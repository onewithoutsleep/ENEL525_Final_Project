"""Use saved models to predict volatility for the next trading session."""

import argparse

import joblib
import numpy as np
import sklearn

from data import build_features, load_daily_data
from market_features import load_market_prices
from volatility_features import RECIPE, feature_options, history_rows_needed


def predict_next_day(data_path, model_path, market_data=None):
    # Only load model files you created or trust: joblib files can execute code.
    saved = joblib.load(model_path)
    metadata = saved["metadata"]
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("Use the scikit-learn version in training_info.json, or retrain.")
    daily = load_daily_data(data_path)
    context = metadata.get("market_context")
    extra_options = {}
    ohlc_mode = context and context.get("recipe") == RECIPE
    if ohlc_mode:
        needed = history_rows_needed(context)
        if len(daily) < needed:
            raise ValueError(f"Supply at least {needed} dated target sessions for the OHLC volatility windows.")
        extra_options = feature_options(context, market_data or context["data_directory"])
        latest_required_date = daily["date"].iloc[-1 - context["delay_sessions"]]
        for symbol, prices in extra_options["ohlc_prices"].items():
            if prices.index.min() > daily["date"].iloc[-needed] or prices.index.max() < latest_required_date:
                raise ValueError(f"{symbol} needs dated OHLC history covering the full volatility window.")
    elif context:
        if context.get("recipe") != "log_close_plus_lagged_log_returns_v1":
            raise ValueError("Unknown saved market-input recipe; use the matching code version.")
        prices = load_market_prices(market_data or context["data_directory"], context["symbols"])
        extra_options = {
            "market_prices": prices, "market_lookback_days": context["lookback_days"],
            "market_delay_sessions": context["delay_sessions"],
        }
    elif market_data is not None:
        raise ValueError("This model was trained without market inputs. Retrain with --market-data first.")
    features = build_features(
        daily, metadata["lookback_days"], metadata["baseline_days"], **extra_options
    )
    expected = metadata["feature_names"]
    if list(features.columns) != expected:
        raise ValueError("The feature recipe changed. Restore the original code or retrain.")
    latest_inputs = features.iloc[[-1]][expected]
    required = [name for name in expected if not (ohlc_mode and name.startswith("market_"))]
    if not np.isfinite(latest_inputs[required].to_numpy()).all():
        raise ValueError("Supply enough complete target and ETF price history to fill every input window.")

    baseline_window = daily["realized_volatility"].tail(metadata["baseline_days"])
    if len(baseline_window) < metadata["baseline_days"] or baseline_window.isna().any():
        raise ValueError("Supply a complete target volatility history for the baseline.")
    predictions = {
        metadata["baseline_name"]: float(baseline_window.mean())
    }
    for name, model in saved["models"].items():
        value = float(model.predict(latest_inputs)[0])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid prediction from {name}; check the inputs and settings.")
        predictions[name] = value

    as_of = daily["date"].iloc[-1].date()
    print(f"Using completed data through {as_of}; forecast is for the NEXT trading session.")
    print("Volatility is a size of price movement, not a price or a direction.")
    for name, value in predictions.items():
        selected = " (selected on validation data)" if name == metadata["selected_model_by_validation"] else ""
        print(f"{name}: {value:.6f} = {value:.3%}{selected}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="History including the latest COMPLETE session")
    parser.add_argument("--models", default="results/models.joblib")
    parser.add_argument("--market-data", help="Override the saved price or OHLC folder after moving/updating data")
    args = parser.parse_args()
    try:
        predict_next_day(args.data, args.models, market_data=args.market_data)
    except (ValueError, OSError) as error:
        parser.exit(1, f"\nCannot predict: {error}\n")


if __name__ == "__main__":
    main()
