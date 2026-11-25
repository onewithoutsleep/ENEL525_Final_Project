"""Read one stock's daily data and build inputs using only known history."""

import numpy as np
import pandas as pd

from config import BASELINE_DAYS, LOOKBACK_DAYS
from volatility_features import build_ohlc_features, history_rows_needed


def load_daily_data(path):
    """One row per trading day; volatility is a decimal (0.02 means 2%)."""
    daily = pd.read_csv(path)
    required = ["date", "close", "volume", "realized_volatility"]
    missing = [column for column in required if column not in daily.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    if daily.empty:
        raise ValueError("The CSV is empty. Add real data, or run make_demo_data.py.")
    if "symbol" in daily.columns and daily["symbol"].nunique(dropna=False) != 1:
        raise ValueError("Use one stock per file. Do not mix different symbols.")

    # Extra CSV columns are deliberately NOT treated as inputs automatically:
    # a column containing tomorrow's information would leak the answer.
    optional = ["session_complete"] if "session_complete" in daily else []
    daily = daily[required + optional].copy()
    daily["date"] = pd.to_datetime(daily["date"], format="%Y-%m-%d", errors="raise")
    if daily["date"].isna().any():
        raise ValueError("Every row needs a date in YYYY-MM-DD format.")
    if daily["date"].duplicated().any():
        raise ValueError("Duplicate dates found. Use exactly one row per trading day.")
    daily = daily.sort_values("date").reset_index(drop=True)

    numeric_columns = required[1:]
    for column in numeric_columns:
        daily[column] = pd.to_numeric(daily[column], errors="raise")
    complete = pd.Series(True, index=daily.index)
    if "session_complete" in daily:
        flags = pd.to_numeric(daily["session_complete"], errors="raise")
        if not flags.isin([0, 1]).all():
            raise ValueError("session_complete must contain only 0 or 1.")
        complete = flags == 1
        # Preserve the missing session's position. Never make the next available
        # day look like the immediately following session by dropping this row.
        daily[numeric_columns] = daily[numeric_columns].astype(float)
        daily.loc[~complete, numeric_columns] = np.nan
    if not np.isfinite(daily.loc[complete, numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Prices, volume, and volatility must be finite, with no blanks.")
    if (daily["close"] <= 0).any():
        raise ValueError("Closing prices must be greater than zero.")
    if (daily[["volume", "realized_volatility"]] < 0).any().any():
        raise ValueError("Volume and realized volatility cannot be negative.")
    return daily


def build_features(daily, lookback_days=LOOKBACK_DAYS, baseline_days=BASELINE_DAYS,
                   market_prices=None, market_lookback_days=20, market_delay_sessions=1,
                   ohlc_prices=None, ohlc_context=None):
    """At the close of day t, use information up to t to predict day t+1."""
    if lookback_days < 1 or baseline_days < 1:
        raise ValueError("Lookback and baseline windows must be positive integers.")

    # A log return measures the change in price from yesterday to today.
    daily_return = np.log(daily["close"]).diff()
    log_volume = np.log1p(daily["volume"])
    volatility = daily["realized_volatility"]
    # Collect columns first so larger windows (such as 60 days) stay efficient.
    features = {}

    # lag_0 is today, lag_1 is yesterday, etc. All are known at today's close.
    for lag in range(lookback_days):
        features[f"return_lag_{lag}"] = daily_return.shift(lag)
        if ohlc_context is None:
            features[f"volatility_lag_{lag}"] = volatility.shift(lag)
        features[f"log_volume_lag_{lag}"] = log_volume.shift(lag)

    features["log_close"] = np.log(daily["close"])
    if ohlc_context is None:
        features["volatility_mean_5"] = volatility.rolling(5).mean()
        features["volatility_mean_baseline"] = volatility.rolling(baseline_days).mean()
    features["relative_log_volume"] = log_volume - log_volume.rolling(baseline_days).mean()

    # A repeating calendar representation avoids treating the date as a price.
    weekday = daily["date"].dt.dayofweek
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result = pd.DataFrame(features, index=daily.index)
    if market_prices is not None:
        from market_features import build_market_features
        extra = build_market_features(daily, market_prices, market_lookback_days, market_delay_sessions)
        result = pd.concat([result, extra], axis=1)
    if ohlc_context is not None:
        extra = build_ohlc_features(
            daily, ohlc_prices, ohlc_context["target_symbol"], ohlc_context["lookback_days"],
            ohlc_context["delay_sessions"], ohlc_context["volatility_days"],
            ohlc_context["horizons"], ohlc_context["minimum_coverage"],
        )
        result = pd.concat([result, extra], axis=1)
    return result


def make_training_table(daily, lookback_days=LOOKBACK_DAYS, baseline_days=BASELINE_DAYS,
                        market_prices=None, market_lookback_days=20, market_delay_sessions=1,
                        ohlc_prices=None, ohlc_context=None):
    features = build_features(daily, lookback_days, baseline_days,
                              market_prices, market_lookback_days, market_delay_sessions,
                              ohlc_prices, ohlc_context)
    feature_names = list(features.columns)
    table = features.copy()
    table["forecast_date"] = daily["date"]
    table["target_date"] = daily["date"].shift(-1)
    table["target"] = daily["realized_volatility"].shift(-1)
    table["baseline"] = daily["realized_volatility"].rolling(baseline_days).mean()

    # Early rows lack enough history; the last row has no known future answer.
    # Explicitly flagged incomplete sessions also invalidate affected historical
    # windows and targets. Dates are kept until AFTER shifts and rolling windows.
    if ohlc_context is None:
        table = table.dropna()
    else:
        # Require a full calendar window and complete target inputs/answer.
        # Missing outside-market features are handled by each fitted model.
        required = [name for name in table if not name.startswith("market_")]
        table = table.iloc[history_rows_needed(ohlc_context) - 1:].dropna(subset=required)
    table = table.reset_index(drop=True)
    return table, feature_names


def chronological_split(table, train_fraction=0.60, validation_fraction=0.20):
    """Split oldest to newest, leaving a one-row gap at each boundary."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than 1.")
    train_end = int(len(table) * train_fraction)
    validation_end = int(len(table) * (train_fraction + validation_fraction))
    train = table.iloc[:train_end].copy()
    validation = table.iloc[train_end + 1:validation_end].copy()
    test = table.iloc[validation_end + 1:].copy()
    if min(len(train), len(validation), len(test)) < 2:
        raise ValueError("Not enough examples for three separate date ranges.")

    # The gaps keep training answers strictly before validation input dates,
    # and validation answers strictly before test input dates.
    assert train["target_date"].max() < validation["forecast_date"].min()
    assert validation["target_date"].max() < test["forecast_date"].min()
    return train, validation, test
