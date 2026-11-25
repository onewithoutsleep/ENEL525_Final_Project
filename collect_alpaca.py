"""Download 5-minute bars and calculate daily inputs for several stocks/ETFs."""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpaca_access import AlpacaReader, load_credentials
from data import load_daily_data, make_training_table

ROOT = Path(__file__).resolve().parent
MARKET_TIMEZONE = "America/New_York"


def download_bars(client, symbol, start, end, feed, mapping_date=None, page_size=10000,
                  timeframe="5Min"):
    """Follow every page token. A single response is NOT the whole history."""
    first = pd.Timestamp(start, tz=MARKET_TIMEZONE).tz_convert("UTC")
    last = (pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize(MARKET_TIMEZONE).tz_convert("UTC")
    last -= pd.Timedelta(seconds=1)  # Alpaca's end time is inclusive.
    parameters = {
        "timeframe": timeframe, "start": first.isoformat(), "end": last.isoformat(),
        "feed": feed, "adjustment": "split", "limit": page_size, "sort": "asc",
        "asof": mapping_date or end,
    }
    rows = []
    tokens_seen = set()
    while True:
        result = client.get("data.alpaca.markets", f"/v2/stocks/{symbol}/bars", parameters)
        rows.extend(result.get("bars") or [])
        token = result.get("next_page_token")
        if not token:
            break
        if token in tokens_seen:
            raise ValueError("Alpaca repeated a page token; download stopped to prevent duplicate data.")
        tokens_seen.add(token)
        parameters["page_token"] = token
    if not rows:
        # A symbol may have no history in an early year; keep that absence
        # visible in the session audit rather than inventing earlier prices.
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    bars = pd.DataFrame(rows).rename(columns={
        "t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "n": "trade_count", "vw": "vwap",
    })
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(bars.columns):
        raise ValueError("Alpaca returned an unexpected bar format.")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    return bars.sort_values("timestamp").reset_index(drop=True)


def aggregate_daily(bars, calendar):
    """Require every regular-session 5-minute interval, including early closes."""
    # Work with whole columns rather than thousands of small daily tables.
    # This produces the same calculations much faster for a decade of history.
    schedule = pd.DataFrame(calendar)[["date", "open", "close"]].copy()
    for column in ["open", "close"]:
        schedule[column] = pd.to_datetime(
            schedule["date"] + " " + schedule[column], format="mixed"
        ).dt.tz_localize(MARKET_TIMEZONE).dt.tz_convert("UTC")
    schedule = schedule.rename(columns={"open": "session_open", "close": "session_close"})
    schedule = schedule.set_index("date")

    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    if bars["timestamp"].isna().any():
        raise ValueError("A bar has no usable timestamp.")
    bars["session_date"] = bars["timestamp"].dt.tz_convert(MARKET_TIMEZONE).dt.strftime("%Y-%m-%d")
    bars = bars.join(schedule, on="session_date")
    regular = bars.loc[(bars["timestamp"] >= bars["session_open"])
                       & (bars["timestamp"] < bars["session_close"])].copy()
    regular = regular.sort_values(["session_date", "timestamp"])
    offset = (regular["timestamp"] - regular["session_open"]).dt.total_seconds()
    regular["on_grid"] = (offset % 300) == 0
    regular["duplicate"] = regular.duplicated(["session_date", "timestamp"])
    numeric = regular[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    regular["valid_values"] = (
        np.isfinite(numeric).all(axis=1)
        & (numeric[:, :4] > 0).all(axis=1) & (numeric[:, 4] >= 0)
        & (numeric[:, 1] >= numeric[:, [0, 2, 3]].max(axis=1))
        & (numeric[:, 2] <= numeric[:, [0, 1, 3]].min(axis=1))
    )
    by_day = regular.groupby("session_date", sort=False)
    previous_close = by_day["close"].shift(1)
    previous_price = previous_close.fillna(regular["open"])
    # At each session's first row, use its open instead of yesterday's close.
    with np.errstate(divide="ignore", invalid="ignore"):
        regular["squared_return"] = (np.log(regular["close"]) - np.log(previous_price)) ** 2

    audit = pd.DataFrame(index=schedule.index)
    audit["expected_bars"] = ((schedule["session_close"] - schedule["session_open"])
                              .dt.total_seconds() / 300).astype(int)
    audit["received_regular_bars"] = by_day.size().reindex(audit.index, fill_value=0)
    unique = regular.drop_duplicates(["session_date", "timestamp"])
    aligned_counts = unique.loc[unique["on_grid"]].groupby("session_date").size()
    unexpected_counts = unique.loc[~unique["on_grid"]].groupby("session_date").size()
    audit["missing_bars"] = audit["expected_bars"] - aligned_counts.reindex(audit.index, fill_value=0)
    audit["unexpected_bars"] = unexpected_counts.reindex(audit.index, fill_value=0)
    audit["duplicate_bars"] = by_day["duplicate"].sum().reindex(audit.index, fill_value=0)
    audit["valid_values"] = by_day["valid_values"].all().reindex(audit.index, fill_value=True)
    audit["session_complete"] = (
        (audit[["missing_bars", "unexpected_bars", "duplicate_bars"]] == 0).all(axis=1)
        & audit["valid_values"] & (audit["received_regular_bars"] == audit["expected_bars"])
    )
    daily = pd.DataFrame(index=schedule.index)
    daily["close"] = by_day["close"].last()
    daily["volume"] = by_day["volume"].sum()
    daily["realized_volatility"] = np.sqrt(by_day["squared_return"].sum())
    daily.loc[~audit["session_complete"], :] = np.nan
    daily["session_complete"] = audit["session_complete"].astype(int)
    return daily.reset_index(), audit.reset_index()


def collect_one_symbol(client, symbol, start, end, output, feed, calendar, page_size=10000):
    """Cache each year separately so an interruption does not lose earlier years."""
    frames = []
    raw_folder = output / "raw_5minute" / symbol
    raw_folder.mkdir(parents=True, exist_ok=True)
    try:
        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            first = max(start, f"{year}-01-01")
            last = min(end, f"{year}-12-31")
            raw_path = raw_folder / f"{year}.csv.gz"
            if raw_path.exists():
                bars = pd.read_csv(raw_path)
            else:
                bars = download_bars(client, symbol, first, last, feed, mapping_date=end, page_size=page_size)
                partial = raw_path.with_suffix(".partial")
                bars.to_csv(partial, index=False, compression="gzip")
                partial.replace(raw_path)
            frames.append(bars)
            print(f"  {symbol} {year}: {len(bars):,} bars saved", flush=True)
        bars = pd.concat(frames, ignore_index=True)
        if bars.empty:
            raise ValueError("No bars returned anywhere in the requested date range.")
        daily, quality = aggregate_daily(bars, calendar)
        daily_path = output / "daily" / f"{symbol}.csv"
        daily.to_csv(daily_path, index=False)
        quality.to_csv(output / "quality" / f"{symbol}.csv", index=False)
        training_table, _ = make_training_table(load_daily_data(daily_path))
        complete_dates = daily.loc[daily["session_complete"] == 1, "date"]
        item = {"symbol": symbol, "status": "ok", "raw_bars": len(bars),
                "calendar_sessions": len(daily),
                "complete_sessions": int(daily["session_complete"].sum()),
                "incomplete_sessions": int((daily["session_complete"] == 0).sum()),
                "usable_training_examples": len(training_table),
                "first_date": daily["date"].iloc[0], "last_date": daily["date"].iloc[-1],
                "first_complete_date": complete_dates.min() if len(complete_dates) else None,
                "last_complete_date": complete_dates.max() if len(complete_dates) else None}
        print(f"{symbol} finished: {item['complete_sessions']} complete sessions; {len(training_table)} usable examples", flush=True)
        return item
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        print(f"Could not finish {symbol}: {error}", flush=True)
        return {"symbol": symbol, "status": "failed", "error": str(error)}


def collect(symbols_path, start, end, output, feed="sip", workers=3, page_size=10000):
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date > end_date:
        raise ValueError("Start date must be before the end date.")
    today = pd.Timestamp.now(tz=MARKET_TIMEZONE).date()
    if end_date.date() >= today:
        raise ValueError("Use an end date before today so only completed sessions are collected.")
    symbols = pd.read_csv(symbols_path)
    if "symbol" not in symbols or symbols.empty:
        raise ValueError("The symbol list must contain a nonempty 'symbol' column.")
    symbols["symbol"] = symbols["symbol"].str.upper()
    if symbols["symbol"].duplicated().any() or not symbols["symbol"].map(
        lambda value: bool(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", str(value)))
    ).all():
        raise ValueError("Invalid or duplicated ticker symbol.")
    client = AlpacaReader(*load_credentials())
    calendar = client.calendar(start, end)
    if not isinstance(calendar, list) or not calendar:
        raise ValueError("Alpaca did not return a usable trading calendar.")
    calendar = sorted(calendar, key=lambda row: row["date"])
    if len({row["date"] for row in calendar}) != len(calendar):
        raise ValueError("The trading calendar has duplicate dates.")

    output = Path(output)
    request_info = {"source": "Alpaca", "start": start, "end": end,
                    "feed": feed, "interval": "5Min", "adjustment": "split"}
    manifest_path = output / "collection_info.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["request"] != request_info:
            raise ValueError("This folder contains different collection settings. Choose a new folder.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output folder, or a folder with a matching collection manifest.")
    for folder in ["raw_5minute", "daily", "quality"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    manifest = {"request": request_info, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "symbols": symbols["symbol"].tolist(),
                "calendar_sessions": len(calendar), "status": "in_progress",
                "volatility_definition": "sqrt(sum of squared intraday log returns); first open included; overnight excluded",
                "missing_data_policy": "retain calendar row; mark incomplete; exclude affected input/target windows",
                "universe_note": "Hand-picked current symbols, not a historical point-in-time universe."}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(calendar).to_csv(output / "market_calendar.csv", index=False)
    symbols.to_csv(output / "symbols.csv", index=False)
    # On a restart, keep completed symbols and retry only unfinished ones.
    summary_path = output / "collection_summary.csv"
    summary = []
    if summary_path.exists():
        previous_summary = pd.read_csv(summary_path)
        for item in previous_summary.to_dict("records"):
            symbol = item["symbol"]
            if (symbol in set(symbols["symbol"]) and item["status"] == "ok"
                    and (output / "daily" / f"{symbol}.csv").exists()
                    and (output / "quality" / f"{symbol}.csv").exists()):
                summary.append(item)
    completed_symbols = {item["symbol"] for item in summary}
    if completed_symbols:
        print(f"Reusing {len(completed_symbols)} completed symbols.", flush=True)
    # A few symbols can download at once, while the shared reader enforces one
    # combined request limit. Set --workers 1 for entirely sequential execution.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(collect_one_symbol, client, symbol, start, end, output, feed, calendar, page_size)
                   for symbol in symbols["symbol"] if symbol not in completed_symbols]
        for finished in as_completed(pending):
            summary.append(finished.result())
            pd.DataFrame(summary).sort_values("symbol").to_csv(output / "collection_summary.csv", index=False)
            print(f"Completed {len(summary)}/{len(symbols)} symbols.", flush=True)
    pd.DataFrame(summary).sort_values("symbol").to_csv(summary_path, index=False)
    failures = sum(item["status"] != "ok" for item in summary)
    manifest["status"] = "complete" if failures == 0 else "complete_with_failures"
    manifest["failed_symbols"] = failures
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Finished: {len(summary) - failures}/{len(summary)} symbols. Saved to {output.resolve()}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=str(ROOT / "symbols.csv"))
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-11-09")
    parser.add_argument("--output", default=str(ROOT / "data" / "alpaca_since2016"))
    parser.add_argument("--feed", choices=["sip", "iex"], default="sip")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--page-size", type=int, choices=[1000, 5000, 10000], default=10000,
                        help="Try 1000 if large requests repeatedly time out.")
    args = parser.parse_args()
    try:
        failures = collect(args.symbols, args.start, args.end, args.output, args.feed, args.workers, args.page_size)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Cannot collect: {error}\n")
    if failures:
        parser.exit(2, "Some symbols failed; see collection_summary.csv and rerun to resume.\n")


if __name__ == "__main__":
    main()
