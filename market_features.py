"""Add other ETFs' historical prices without using future information."""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_market_prices(directory, symbols):
    """Read one date/close CSV per ETF. Missing sessions are never filled."""
    prices = {}
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Choose a nonempty list of distinct market symbols.")
    for symbol in symbols:
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol):
            raise ValueError(f"Invalid market symbol: {symbol}")
        path = Path(directory) / f"{symbol}.csv"
        frame = pd.read_csv(path)
        if not {"date", "close"}.issubset(frame.columns) or frame.empty:
            raise ValueError(f"{path.name} needs nonempty date and close columns.")
        frame = frame[["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
        if frame["date"].isna().any() or frame["date"].duplicated().any():
            raise ValueError(f"{path.name} has missing or duplicate dates.")
        frame["close"] = pd.to_numeric(frame["close"], errors="raise")
        observed = frame["close"].dropna()
        if observed.empty or not np.isfinite(observed).all() or (observed <= 0).any():
            raise ValueError(f"{path.name} needs finite, positive observed prices.")
        prices[symbol] = frame.sort_values("date").set_index("date")["close"]
    return prices


def build_market_features(daily, market_prices, lookback_days=20, delay_sessions=1):
    """For each ETF: one log price plus a separate return for each history day.

    Align by date FIRST, then calculate returns and shifts. If a price/date is
    absent, it stays missing; we never turn a two-day change into a one-day one.
    delay=1 means a forecast after Tuesday's close uses ETF prices through Monday.
    """
    if not isinstance(lookback_days, int) or lookback_days < 1:
        raise ValueError("Market history length must be a positive integer.")
    if not isinstance(delay_sessions, int) or delay_sessions < 1:
        raise ValueError("Finalized daily ETF bars need at least a one-session delay.")
    columns = {}
    dates = pd.DatetimeIndex(daily["date"])
    for symbol, prices in market_prices.items():
        # Reindex to the TARGET's complete exchange calendar, including bad days.
        aligned = pd.Series(prices.reindex(dates).to_numpy(), index=daily.index)
        log_price = np.log(aligned)
        returns = log_price.diff()
        columns[f"market_{symbol}_log_close"] = log_price.shift(delay_sessions)
        for lag in range(lookback_days):
            columns[f"market_{symbol}_return_lag_{lag}"] = returns.shift(delay_sessions + lag)
    return pd.DataFrame(columns, index=daily.index)


def split_using_saved_dates(table, boundaries):
    """Keep an earlier experiment's calendar boundaries despite new missing rows."""
    pieces = []
    previous_target_end = None
    for name in ["train", "validation", "test"]:
        span = boundaries[name]
        start = pd.Timestamp(span["first_forecast_date"])
        end = pd.Timestamp(span["last_forecast_date"])
        target_start = pd.Timestamp(span["first_target_date"])
        target_end = pd.Timestamp(span["last_target_date"])
        if not (start <= end < target_end and start < target_start <= target_end):
            raise ValueError(f"Invalid saved date boundaries for {name}.")
        if previous_target_end is not None and previous_target_end >= start:
            raise ValueError("Saved split dates overlap future labels.")
        piece = table.loc[
            table["forecast_date"].between(start, end)
            & table["target_date"].between(target_start, target_end)
        ].copy()
        if len(piece) < 2:
            raise ValueError(f"Too few {name} examples after market-input exclusions.")
        pieces.append(piece)
        previous_target_end = target_end
    return tuple(pieces)
