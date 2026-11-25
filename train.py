"""Train and validate models; reserve the final test unless --evaluate-test is set."""

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import config
from data import chronological_split, load_daily_data, make_training_table
from metrics import evaluate_predictions, improvement_percent
from models import build_decision_tree, build_neural_network, build_random_forest
from market_features import load_market_prices, split_using_saved_dates
from volatility_features import RECIPE, VOLATILITY_DAYS, HORIZONS, MIN_COVERAGE, feature_options


def describe_date_range(split):
    return {
        "rows": len(split),
        "first_forecast_date": str(split["forecast_date"].min().date()),
        "last_forecast_date": str(split["forecast_date"].max().date()),
        "first_target_date": str(split["target_date"].min().date()),
        "last_target_date": str(split["target_date"].max().date()),
    }


def train_models(data_path, output_dir, tune=False, evaluate_test=False,
                 market_data=None, market_symbols=None, target_symbol=None, split_from=None,
                 volatility_data=None):
    """Use validation for development. Test evaluation must be requested explicitly."""
    daily = load_daily_data(data_path)
    table, feature_names = make_training_table(daily)
    if len(table) < config.MINIMUM_EXAMPLES:
        raise ValueError(
            f"Only {len(table)} usable examples. This starter requires at least "
            f"{config.MINIMUM_EXAMPLES} after lookback/target preparation. "
            "Use more history, or run the synthetic demo."
        )
    # Establish dates BEFORE external-price gaps can remove examples. An existing
    # experiment's dates take priority so adding inputs cannot expose its test.
    original_splits = chronological_split(table, config.TRAIN_FRACTION, config.VALIDATION_FRACTION)
    boundaries = {name: describe_date_range(part) for name, part in
                  zip(["train", "validation", "test"], original_splits)}
    if split_from is not None:
        reference = json.loads(Path(split_from).read_text(encoding="utf-8"))
        boundaries = reference.get("split_boundaries") or reference["date_ranges"]
    market_context = None
    if market_data is not None and volatility_data is not None:
        raise ValueError("Choose --volatility-data or --market-data. The OHLC mode already includes prices.")
    if market_data is not None:
        target_symbol = (target_symbol or Path(data_path).stem).upper()
        symbols = [symbol.upper() for symbol in
                   (market_symbols if market_symbols is not None else config.MARKET_SYMBOLS)
                   if symbol.upper() != target_symbol]
        prices = load_market_prices(market_data, symbols)
        table, feature_names = make_training_table(
            daily, market_prices=prices, market_lookback_days=config.MARKET_LOOKBACK_DAYS,
            market_delay_sessions=config.MARKET_DELAY_SESSIONS,
        )
        fingerprints = {}
        for symbol in symbols:
            with (Path(market_data) / f"{symbol}.csv").open("rb") as stream:
                fingerprints[symbol] = hashlib.file_digest(stream, "sha256").hexdigest()
        market_context = {
            "data_directory": str(Path(market_data).resolve()), "symbols": symbols,
            "target_symbol": target_symbol,
            "lookback_days": config.MARKET_LOOKBACK_DAYS,
            "delay_sessions": config.MARKET_DELAY_SESSIONS,
            "recipe": "log_close_plus_lagged_log_returns_v1",
            "price_definition": "Alpaca split-adjusted daily ETF close; not spot commodity prices",
            "missing_policy": "Align to target calendar first; no filling; exclude affected examples",
            "file_sha256": fingerprints,
        }
    elif volatility_data is not None:
        target_symbol = (target_symbol or Path(data_path).stem).upper()
        folder = Path(volatility_data)
        if market_symbols is None:
            manifest = json.loads((folder / "collection_info.json").read_text(encoding="utf-8"))
            market_symbols = manifest["request"]["symbols"]
        symbols = [symbol.upper() for symbol in market_symbols]
        if target_symbol not in symbols:
            symbols.append(target_symbol)
        market_context = {
            "data_directory": str(folder.resolve()), "symbols": symbols,
            "target_symbol": target_symbol, "recipe": RECIPE,
            "lookback_days": config.MARKET_LOOKBACK_DAYS,
            "delay_sessions": config.MARKET_DELAY_SESSIONS,
            "volatility_days": VOLATILITY_DAYS, "horizons": HORIZONS,
            "minimum_coverage": MIN_COVERAGE,
            "target_definition": "next-session five-minute intraday realized volatility",
            "price_definition": "Alpaca SIP split-adjusted daily OHLC",
            "missing_policy": "Apply rolling coverage and fit median imputer on training inputs",
        }
        options = feature_options(market_context, folder)
        table, feature_names = make_training_table(daily, **options)
        fingerprints = {}
        for symbol in symbols:
            with (folder / f"{symbol}.csv").open("rb") as stream:
                fingerprints[symbol] = hashlib.file_digest(stream, "sha256").hexdigest()
        market_context["file_sha256"] = fingerprints
    elif market_symbols is not None:
        raise ValueError("--market-symbols needs --market-data or --volatility-data.")
    if market_context and len(table) < config.MINIMUM_EXAMPLES:
        raise ValueError(
            f"Only {len(table)} usable examples after adding market inputs. "
            f"At least {config.MINIMUM_EXAMPLES} are required. "
            "Supply more overlapping history or choose fewer ETFs."
        )
    if market_context or split_from is not None:
        train, validation, test = split_using_saved_dates(table, boundaries)
    else:
        train, validation, test = original_splits
    output_dir = Path(output_dir)
    # Preserve earlier experiments by requiring a new output folder each run.
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Inputs per example: {len(feature_names)}")
    print(f"Training: {len(train)} | validation: {len(validation)} | test: {len(test)}")
    if not evaluate_test:
        print("Validation-only run: the final test period will not be scored.")

    model_choices = {
        "decision_tree": (build_decision_tree, config.TREE_SETTINGS),
        "random_forest": (build_random_forest, config.FOREST_SETTINGS),
        "neural_network": (build_neural_network, config.NETWORK_SETTINGS),
    }
    fitted_models = {}
    chosen_settings = {}
    candidate_results = []
    validation_errors = {}
    convergence_notes = []
    baseline_name = f"baseline_{config.BASELINE_DAYS}_day"

    # 1. Fit only on the earliest date range.
    # 2. Use the middle date range to choose settings, if --tune is enabled.
    for name, (build_model, settings_list) in model_choices.items():
        best_error = float("inf")
        for candidate_number, settings in enumerate(settings_list if tune else settings_list[:1], 1):
            print(f"Training {name}, candidate {candidate_number}...", flush=True)
            model = build_model(settings)
            if volatility_data is not None:
                # Fit substitutions on training data only. Keep a flag when a
                # value was absent, so missing data is not a real observation.
                model = Pipeline([
                    ("fill_missing_markets", SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True,
                    )),
                    ("regressor", model),
                ])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train[feature_names], train["target"])
            for warning in caught:
                note = f"{name}, candidate {candidate_number}: {warning.message}"
                convergence_notes.append(note)
                print(f"Training warning: {note}")

            prediction = model.predict(validation[feature_names])
            errors = evaluate_predictions(validation["target"], prediction)
            candidate_results.append({
                "model": name, "candidate": candidate_number,
                "settings": json.dumps(settings), **errors,
            })
            if errors["QLIKE"] < best_error:
                best_error = errors["QLIKE"]
                fitted_models[name] = model
                chosen_settings[name] = settings
        validation_errors[name] = best_error

    # Select before looking at test scores. Include the simple baseline:
    # a complicated model is not automatically better.
    validation_errors[baseline_name] = evaluate_predictions(
        validation["target"], validation["baseline"]
    )["QLIKE"]
    selected_model = min(validation_errors, key=validation_errors.get)

    # 3. Save validation results. Only evaluate test data when explicitly asked.
    # Each row is a new ONE-DAY forecast using history then available.
    # No model or scaler is refitted using validation or test answers.
    score_rows = []
    splits_to_evaluate = [("validation", validation)]
    if evaluate_test:
        splits_to_evaluate.append(("test", test))
    for split_name, split in splits_to_evaluate:
        predictions = split[["forecast_date", "target_date", "target", "baseline"]].copy()
        baseline_errors = evaluate_predictions(split["target"], split["baseline"])
        score_rows.append({
            "split": split_name, "model": baseline_name, **baseline_errors,
            "QLIKE_improvement_percent": 0.0, "meets_5_percent_goal": False,
        })
        for name, model in fitted_models.items():
            prediction = model.predict(split[feature_names])
            errors = evaluate_predictions(split["target"], prediction)
            improvement = improvement_percent(baseline_errors["QLIKE"], errors["QLIKE"])
            score_rows.append({
                "split": split_name, "model": name, **errors,
                "QLIKE_improvement_percent": improvement,
                "meets_5_percent_goal": improvement is not None and improvement >= 5,
            })
            predictions[name] = prediction
        predictions.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    # Save model + feature recipe + settings together so predictions use the
    # exact same input columns and scaling as training.
    metadata = {
        "data_file": str(Path(data_path).resolve()),
        "market_context": market_context,
        "split_boundaries": boundaries,
        "split_reference_file": str(Path(split_from).resolve()) if split_from else None,
        "evaluation_mode": "validation_and_test" if evaluate_test else "validation_only",
        "test_evaluated": evaluate_test,
        "tuning_enabled": tune,
        "random_seed": config.RANDOM_SEED,
        "train_fraction": config.TRAIN_FRACTION,
        "validation_fraction": config.VALIDATION_FRACTION,
        "lookback_days": config.LOOKBACK_DAYS,
        "baseline_days": config.BASELINE_DAYS,
        "baseline_name": baseline_name,
        "feature_names": feature_names,
        "selected_model_by_validation": selected_model,
        "chosen_settings": chosen_settings,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "convergence_notes": convergence_notes,
        "training_last_target_date": str(train["target_date"].max().date()),
        "date_ranges": {},
    }
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        metadata["date_ranges"][name] = describe_date_range(split)
    joblib.dump({"models": fitted_models, "metadata": metadata}, output_dir / "models.joblib")
    (output_dir / "training_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(candidate_results).to_csv(output_dir / "validation_candidates.csv", index=False)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(output_dir / "scores.csv", index=False)
    print("\nValidation results (lower error is better):")
    print(scores[scores["split"] == "validation"].to_string(index=False))
    if evaluate_test:
        print("\nFinal test results (lower error is better):")
        print(scores[scores["split"] == "test"].to_string(index=False))
    print(f"\nSelected using validation only: {selected_model}")
    print(f"Saved results to: {output_dir.resolve()}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with daily history for one stock")
    parser.add_argument("--output", default="results", help="A NEW folder for this experiment")
    parser.add_argument("--tune", action="store_true", help="Try all settings listed in config.py")
    parser.add_argument("--market-data", help="Folder of daily ETF price CSVs; adds wider-market inputs")
    parser.add_argument("--volatility-data", help="Folder of daily OHLC CSVs; adds prices and volatility inputs")
    parser.add_argument("--market-symbols", nargs="+", help="Optional subset of market symbols")
    parser.add_argument("--target-symbol", help="Symbol being predicted; default is the data filename without .csv")
    parser.add_argument("--split-from", help="Earlier training_info.json whose date boundaries must stay fixed")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument(
        "--validation-only", action="store_true",
        help="Keep the final test unscored (this is the default)",
    )
    evaluation.add_argument(
        "--evaluate-test", action="store_true",
        help="Also score the final test; use only after finishing model development",
    )
    args = parser.parse_args()
    try:
        train_models(args.data, args.output, args.tune, evaluate_test=args.evaluate_test,
                     market_data=args.market_data, market_symbols=args.market_symbols,
                     target_symbol=args.target_symbol, split_from=args.split_from,
                     volatility_data=args.volatility_data)
    except (ValueError, OSError) as error:
        parser.exit(1, f"\nCannot train: {error}\n")


if __name__ == "__main__":
    main()
