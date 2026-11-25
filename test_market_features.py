"""Timing, missing-data, saved-model, and holdout checks for extra ETF inputs."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from collect_market_prices import daily_prices_on_calendar
from data import build_features, chronological_split, make_training_table
from make_demo_data import make_demo_data
from market_features import build_market_features, load_market_prices, split_using_saved_dates
from predict import predict_next_day
from train import describe_date_range, train_models


class MarketFeatureTests(unittest.TestCase):
    def setUp(self):
        self.daily = make_demo_data(220)
        self.daily["date"] = pd.to_datetime(self.daily["date"])
        self.prices = pd.Series(100 * np.exp(np.arange(220) / 500),
                                index=pd.DatetimeIndex(self.daily["date"]), name="close")

    def test_price_dates_and_return_windows_are_delayed(self):
        features = build_market_features(self.daily, {"GLD": self.prices.iloc[::-1]})
        self.assertEqual(features.shape[1], 21)
        self.assertAlmostEqual(features.loc[60, "market_GLD_log_close"], np.log(self.prices.iloc[59]))
        self.assertAlmostEqual(features.loc[60, "market_GLD_return_lag_0"],
                               np.log(self.prices.iloc[59] / self.prices.iloc[58]))
        self.assertAlmostEqual(features.loc[60, "market_GLD_return_lag_19"],
                               np.log(self.prices.iloc[40] / self.prices.iloc[39]))
        self.assertTrue(features.iloc[20].isna().any())
        self.assertTrue(np.isfinite(features.iloc[21]).all())
        with self.assertRaises(ValueError):
            build_market_features(self.daily, {"GLD": self.prices}, delay_sessions=0)

    def test_same_day_and_future_prices_cannot_change_current_prediction_inputs(self):
        before = build_features(self.daily, market_prices={"GLD": self.prices})
        changed = self.prices.copy()
        changed.iloc[100:] *= 9
        after = build_features(self.daily, market_prices={"GLD": changed})
        pd.testing.assert_frame_equal(before.iloc[:101], after.iloc[:101])
        self.assertNotEqual(before.loc[101, "market_GLD_log_close"], after.loc[101, "market_GLD_log_close"])
        self.assertEqual(len(before.columns), 87)

    def test_missing_price_is_not_filled_and_does_not_shift_targets(self):
        missing = self.prices.drop(self.prices.index[50])
        features = build_market_features(self.daily, {"GLD": missing})
        self.assertTrue(np.isfinite(features.iloc[50]).all())
        self.assertTrue(features.iloc[51:72].isna().any(axis=1).all())
        self.assertTrue(np.isfinite(features.iloc[72]).all())
        table, _ = make_training_table(self.daily, market_prices={"GLD": missing})
        for row in table.itertuples():
            position = self.daily.index[self.daily["date"] == row.forecast_date][0]
            self.assertEqual(row.target_date, self.daily.loc[position + 1, "date"])

    def test_fixed_boundaries_do_not_move_after_rows_are_removed(self):
        table, _ = make_training_table(self.daily)
        original = chronological_split(table)
        boundaries = {name: describe_date_range(part) for name, part in
                      zip(["train", "validation", "test"], original)}
        reduced = table.drop(original[0].index[10:35])
        fixed = split_using_saved_dates(reduced, boundaries)
        self.assertEqual(len(fixed[0]), len(original[0]) - 25)
        pd.testing.assert_frame_equal(fixed[1], original[1])
        pd.testing.assert_frame_equal(fixed[2], original[2])
        invalid = json.loads(json.dumps(boundaries))
        invalid["test"]["first_forecast_date"] = boundaries["validation"]["last_target_date"]
        with self.assertRaises(ValueError):
            split_using_saved_dates(table, invalid)

    def test_daily_feed_dates_handle_dst_and_keep_missing_sessions(self):
        calendar = [{"date": date} for date in ["2024-03-08", "2024-03-11", "2024-03-12"]]
        bars = pd.DataFrame({"timestamp": ["2024-03-08T05:00:00Z", "2024-03-12T04:00:00Z"],
                             "close": [100, 102]})
        result = daily_prices_on_calendar(bars, calendar)
        self.assertEqual(result["date"].dt.strftime("%Y-%m-%d").tolist(), [row["date"] for row in calendar])
        self.assertTrue(pd.isna(result.loc[1, "close"]))
        self.assertEqual(result["price_available"].tolist(), [1, 0, 1])
        with self.assertRaises(ValueError):
            daily_prices_on_calendar(pd.concat([bars, bars.iloc[[0]]]), calendar)

    def test_bad_price_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "GLD.csv"
            for prices in [self.prices * -1, self.prices * np.inf,
                           pd.concat([self.prices, self.prices.iloc[[0]]])]:
                prices.rename_axis("date").to_csv(path)
                with self.assertRaises(ValueError):
                    load_market_prices(directory, ["GLD"])
            with self.assertRaises(ValueError):
                load_market_prices(directory, ["../GLD"])

    def test_training_and_prediction_reuse_saved_market_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target_path = folder / "SPY.csv"
            self.daily.to_csv(target_path, index=False)
            self.prices.rename_axis("date").to_csv(folder / "GLD.csv")
            # Real small models exercise serialization, scaling, and prediction.
            with (patch("config.FOREST_SETTINGS", [{"n_estimators": 3, "max_depth": 2}]),
                  contextlib.redirect_stdout(io.StringIO())):
                metadata = train_models(target_path, folder / "models", market_data=folder,
                                        market_symbols=["SPY", "GLD"], target_symbol="SPY")
                result = predict_next_day(target_path, folder / "models" / "models.joblib", market_data=folder)
            self.assertEqual(metadata["market_context"]["symbols"], ["GLD"])
            self.assertEqual(len(metadata["feature_names"]), 87)
            self.assertFalse(metadata["test_evaluated"])
            self.assertFalse((folder / "models" / "test_predictions.csv").exists())
            saved = joblib.load(folder / "models" / "models.joblib")
            prices = load_market_prices(folder, ["GLD"])
            table, names = make_training_table(self.daily, market_prices=prices)
            train, _, _ = split_using_saved_dates(table, metadata["split_boundaries"])
            scaler = saved["models"]["neural_network"].regressor_.named_steps["scale_inputs"]
            np.testing.assert_allclose(scaler.mean_, train[names].mean(), atol=1e-12)
            self.assertEqual(scaler.n_samples_seen_, len(train))
            latest = build_features(self.daily, market_prices=prices).iloc[[-1]]
            for name, model in saved["models"].items():
                self.assertAlmostEqual(result[name], float(model.predict(latest)[0]))
            # A moved folder can be supplied, but an absent required ETF must fail.
            empty_folder = folder / "empty"
            empty_folder.mkdir()
            with self.assertRaises(FileNotFoundError):
                predict_next_day(target_path, folder / "models" / "models.joblib", market_data=empty_folder)


if __name__ == "__main__":
    unittest.main()
