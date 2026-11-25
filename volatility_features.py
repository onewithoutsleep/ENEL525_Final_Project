"""Build daily and multi-session volatility inputs from daily OHLC bars."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from market_features import build_market_features


PRICE_COLUMNS = ["open", "high", "low", "close"]
RECIPE = "ohlc_volatility_v1"
VOLATILITY_DAYS = 10
HORIZONS = (21, 126, 252)
MIN_COVERAGE = 0.90


def load_ohlc(directory, symbols):
    """Read cleaned OHLC files. All four prices may be missing on a bad day."""
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Choose a nonempty list of distinct stock and ETF symbols.")
    result = {}
    for symbol in symbols:
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
            raise ValueError(f"Invalid market symbol: {symbol}")
        path = Path(directory) / f"{symbol}.csv"
        frame = pd.read_csv(path)
        if not set(["date", *PRICE_COLUMNS]).issubset(frame.columns) or frame.empty:
            raise ValueError(f"{path.name} needs date, open, high, low, and close columns.")
        dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        if dates.isna().any() or dates.duplicated().any():
            raise ValueError(f"{path.name} has missing or duplicate dates.")
        prices = frame[PRICE_COLUMNS].apply(pd.to_numeric, errors="raise")
        present = prices.notna().any(axis=1)
        observed = prices.loc[present]
        if (observed.empty or not (np.isfinite(observed) & observed.gt(0)).all().all()
                or not observed["high"].ge(observed[["open", "low", "close"]].max(axis=1)).all()
                or not observed["low"].le(observed[["open", "close"]].min(axis=1)).all()):
            raise ValueError(f"{path.name} has no usable OHLC rows or contains invalid OHLC values.")
        prices.index = pd.DatetimeIndex(dates)
        result[symbol] = prices.sort_index()
    return result


def daily_variance_terms(ohlc):
    """Return the overnight move, intraday move, and Rogers–Satchell variance."""
    log_prices = np.log(ohlc[PRICE_COLUMNS])
    opening = log_prices["open"]
    close = log_prices["close"]
    high = log_prices["high"]
    low = log_prices["low"]
    overnight = opening - close.shift(1)
    intraday = close - opening
    range_variance = (high - opening) * (high - close) + (low - opening) * (low - close)
    # All methods use the same observations. A missing prior close makes the
    # overnight part unavailable, even if the current session has prices.
    valid = overnight.notna() & intraday.notna() & range_variance.notna()
    return overnight.where(valid), intraday.where(valid), range_variance.where(valid)


def yang_zhang_variance(overnight, intraday, range_variance, window, minimum_coverage=1.0):
    """Estimate per-session variance inside a trailing window of >=2 sessions."""
    if window < 2 or not 0 < minimum_coverage <= 1:
        raise ValueError("Yang–Zhang needs at least two sessions and coverage in (0, 1].")
    minimum = max(2, int(np.ceil(window * minimum_coverage)))
    # Count only sessions that have all three component estimates. Use the
    # actual count for the sample variances and for the Yang–Zhang weight.
    valid = overnight.notna() & intraday.notna() & range_variance.notna()
    opening_window = overnight.where(valid).rolling(window, min_periods=minimum)
    intraday_window = intraday.where(valid).rolling(window, min_periods=minimum)
    range_window = range_variance.where(valid).rolling(window, min_periods=minimum)
    n = opening_window.count()
    weight = 0.34 / (1.34 + (n + 1) / (n - 1))
    variance = opening_window.var(ddof=1) + weight * intraday_window.var(ddof=1)
    return (variance + (1 - weight) * range_window.mean()).clip(lower=0)


def build_ohlc_features(daily, by_symbol, target_symbol, price_lookback, delay_sessions,
                        volatility_days, horizons, minimum_coverage):
    """Return price histories plus daily, short, and total volatility inputs."""
    if delay_sessions < 1:
        raise ValueError("Use finalized daily OHLC at least one trading session later.")
    price_histories = {symbol: bars["close"] for symbol, bars in by_symbol.items()
                       if symbol != target_symbol}
    prices = build_market_features(daily, price_histories, price_lookback, delay_sessions)
    dates = pd.DatetimeIndex(daily["date"])
    result = {}
    for symbol, bars in by_symbol.items():
        aligned = bars.reindex(dates).set_axis(daily.index)
        overnight, intraday, range_variance = daily_variance_terms(aligned)
        # A one-day range-based estimate plus that day's overnight jump.
        # It is a separate estimator from multi-session Yang–Zhang.
        daily_variance = overnight.pow(2) + range_variance
        daily_volatility = np.sqrt(daily_variance)
        prefix = f"market_{symbol}"
        for lag in range(volatility_days):
            result[f"{prefix}_daily_volatility_lag_{lag + delay_sessions}"] = daily_volatility.shift(
                delay_sessions + lag
            )
        short_variance = yang_zhang_variance(
            overnight, intraday, range_variance, volatility_days, minimum_coverage
        )
        result[f"{prefix}_yang_zhang_{volatility_days}d"] = np.sqrt(short_variance).shift(delay_sessions)
        for window in horizons:
            minimum = int(np.ceil(window * minimum_coverage))
            avg_daily_variance = daily_variance.rolling(window, min_periods=minimum).mean()
            yz_variance = yang_zhang_variance(overnight, intraday, range_variance, window, minimum_coverage)
            # Period volatility, in decimal units. Scale variance by the
            # number of sessions, then take the square root once.
            result[f"{prefix}_total_range_volatility_{window}d"] = np.sqrt(window * avg_daily_variance).shift(delay_sessions)
            result[f"{prefix}_total_yang_zhang_{window}d"] = np.sqrt(window * yz_variance).shift(delay_sessions)
    volatility = pd.DataFrame(result, index=daily.index)
    return pd.concat([prices, volatility], axis=1)


def feature_options(context, directory):
    """Restore the complete saved recipe for training or prediction."""
    return {
        "ohlc_prices": load_ohlc(directory, context["symbols"]),
        "ohlc_context": context,
    }


def history_rows_needed(context):
    """One preceding close, the full longest window, and the forecast delay."""
    return 1 + context["delay_sessions"] + max(
        context["lookback_days"], context["volatility_days"], *context["horizons"]
    )
