"""Small correctness checks; synthetic data here is never accuracy evidence."""

import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd

from config import LOOKBACK_DAYS, NETWORK_SETTINGS
from data import build_features, chronological_split, load_daily_data, make_training_table
from make_demo_data import make_demo_data
from metrics import evaluate_predictions, improvement_percent
from models import build_neural_network
from train import train_models


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.daily = make_demo_data(220)
        self.daily["date"] = pd.to_datetime(self.daily["date"])

    def test_future_data_does_not_change_past_inputs(self):
        before = build_features(self.daily)
        changed = self.daily.copy()
        changed.loc[120:, ["close", "volume", "realized_volatility"]] *= 4
        after = build_features(changed)
        pd.testing.assert_frame_equal(before.iloc[:120], after.iloc[:120])

    def test_each_target_is_the_next_sessions_volatility(self):
        table, names = make_training_table(self.daily)
        first = table.iloc[0]
        position = self.daily.index[self.daily["date"] == first["forecast_date"]][0]
        self.assertEqual(first["target"], self.daily.loc[position + 1, "realized_volatility"])
        self.assertEqual(first["target_date"], self.daily.loc[position + 1, "date"])
        self.assertEqual(first["volatility_lag_19"], self.daily.loc[position - 19, "realized_volatility"])
        self.assertEqual(len(names), 3 * LOOKBACK_DAYS + 6)
        self.assertNotIn("target", names)
        self.assertEqual(len(table), len(self.daily) - 21)

    def test_date_boundaries_do_not_overlap_labels(self):
        table, _ = make_training_table(self.daily)
        train, validation, test = chronological_split(table)
        self.assertLess(train["target_date"].max(), validation["forecast_date"].min())
        self.assertLess(validation["target_date"].max(), test["forecast_date"].min())

    def test_larger_lookback_and_latest_row_without_a_target(self):
        features = build_features(self.daily, lookback_days=60)
        self.assertEqual(features.shape[1], 186)
        self.assertTrue(np.isfinite(features.iloc[-1]).all())
        self.assertEqual(features.iloc[-1]["volatility_lag_59"], self.daily.iloc[-60]["realized_volatility"])
        minimum_history = build_features(self.daily.tail(21).reset_index(drop=True))
        self.assertTrue(np.isfinite(minimum_history.iloc[-1]).all())

    def test_qlike_uses_variance_and_handles_zero(self):
        metrics = evaluate_predictions([0.02], [0.01])
        self.assertAlmostEqual(metrics["QLIKE"], 4 - np.log(4) - 1)
        self.assertAlmostEqual(metrics["MAE"], 0.01)
        self.assertAlmostEqual(metrics["MSE"], 0.0001)
        self.assertEqual(evaluate_predictions([0, 0.02], [0, 0.02])["QLIKE"], 0)
        self.assertAlmostEqual(improvement_percent(2, 1.8), 10)
        self.assertIsNone(improvement_percent(0, 0))

    def test_bad_csv_is_rejected_and_dates_are_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            self.daily.iloc[::-1].to_csv(path, index=False)
            self.assertTrue(load_daily_data(path)["date"].is_monotonic_increasing)
            bad_frames = [
                self.daily.iloc[:0],
                pd.concat([self.daily, self.daily.iloc[[0]]]),
                self.daily.assign(close=-1),
                self.daily.assign(realized_volatility=np.nan),
                self.daily.assign(symbol=["A", "B"] * 110),
                self.daily.drop(columns="volume"),
            ]
            for bad in bad_frames:
                with self.subTest(columns=list(bad.columns), rows=len(bad)):
                    bad.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_daily_data(path)

    def test_scaler_fits_training_only_and_saved_network_roundtrips(self):
        table, names = make_training_table(self.daily)
        train, _, test = chronological_split(table)
        model = build_neural_network(NETWORK_SETTINGS[0])
        model.fit(train[names], train["target"])
        scaler = model.regressor_.named_steps["scale_inputs"]
        np.testing.assert_allclose(scaler.mean_, train[names].mean().to_numpy(), atol=1e-12)
        self.assertEqual(scaler.n_samples_seen_, len(train))
        before = model.predict(test[names])
        self.assertTrue(np.isfinite(before).all())
        self.assertTrue((before > 0).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.joblib"
            joblib.dump(model, path)
            np.testing.assert_allclose(before, joblib.load(path).predict(test[names]))

    def test_validation_only_never_predicts_or_scores_test_rows(self):
        self.check_evaluation_mode(evaluate_test=False)

    def test_final_test_requires_explicit_opt_in(self):
        # This check uses invented data, so it does not expose the real holdout.
        self.check_evaluation_mode(evaluate_test=True)

    def check_evaluation_mode(self, evaluate_test):
        table, names = make_training_table(self.daily)
        train, validation, test = chronological_split(table)
        allowed_indices = [validation.index]
        if evaluate_test:
            allowed_indices.append(test.index)

        def guarded_prediction(inputs):
            # Fail immediately if any model receives an unauthorized date range.
            self.assertTrue(any(inputs.index.equals(index) for index in allowed_indices))
            return np.full(len(inputs), 0.01)

        def guarded_scores(actual, prediction):
            self.assertTrue(any(actual.index.equals(index) for index in allowed_indices))
            return evaluate_predictions(actual, prediction)

        models = [Mock(predict=Mock(side_effect=guarded_prediction)) for _ in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "synthetic.csv"
            output = Path(directory) / "results"
            self.daily.to_csv(data_path, index=False)
            with (
                patch("train.build_decision_tree", return_value=models[0]),
                patch("train.build_random_forest", return_value=models[1]),
                patch("train.build_neural_network", return_value=models[2]),
                patch("train.evaluate_predictions", side_effect=guarded_scores),
                patch("train.joblib.dump"),  # These guarded models are test doubles.
                contextlib.redirect_stdout(io.StringIO()),
            ):
                # Omitting the option must be safe by default.
                options = {"evaluate_test": True} if evaluate_test else {}
                metadata = train_models(data_path, output, **options)
            for model in models:
                model.fit.assert_called_once()
                pd.testing.assert_frame_equal(model.fit.call_args.args[0], train[names])
                np.testing.assert_allclose(model.fit.call_args.args[1], train["target"])
            self.assertEqual(metadata["test_evaluated"], evaluate_test)
            scores = pd.read_csv(output / "scores.csv")
            expected_splits = {"validation", "test"} if evaluate_test else {"validation"}
            self.assertEqual(set(scores["split"]), expected_splits)
            self.assertEqual((output / "test_predictions.csv").exists(), evaluate_test)
            saved = pd.read_csv(output / "validation_predictions.csv")
            self.assertEqual(len(saved), len(validation))
            np.testing.assert_allclose(saved["target"], validation["target"])
            self.assertEqual(saved["target_date"].iloc[-1], str(validation["target_date"].max().date()))


if __name__ == "__main__":
    unittest.main()
