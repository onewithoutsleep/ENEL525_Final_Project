"""Download split-adjusted daily prices for ETFs and major stocks."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from alpaca_access import AlpacaReader, load_credentials
from collect_alpaca import download_bars, MARKET_TIMEZONE


def bars_on_calendar(bars, calendar, columns):
    """Index daily bars by the New York trading date and keep calendar gaps."""
    dates = pd.DatetimeIndex(pd.to_datetime([row["date"] for row in calendar]), name="date")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("The market calendar must be unique and sorted.")
    timestamps = pd.to_datetime(bars["timestamp"], utc=True)
    bar_dates = timestamps.dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None).dt.normalize()
    if bar_dates.isna().any() or bar_dates.duplicated().any():
        raise ValueError("The daily price feed has missing or duplicate dates.")
    values = bars[list(columns)].apply(pd.to_numeric, errors="raise")
    values.index = pd.DatetimeIndex(bar_dates)
    return values.reindex(dates)


def daily_prices_on_calendar(bars, calendar):
    """Original closing-price export used by the ETF price-only experiment."""
    result = bars_on_calendar(bars, calendar, ["close"])
    prices = result["close"].dropna()
    if not np.isfinite(prices).all() or (prices <= 0).any() or bars["close"].isna().any():
        raise ValueError("Daily prices must be finite and positive.")
    result["price_available"] = result["close"].notna().astype(int)
    return result.reset_index()


def daily_ohlc_on_calendar(bars, calendar):
    columns = ["open", "high", "low", "close"]
    result = bars_on_calendar(bars, calendar, columns + ["volume"])
    prices = result[columns]
    # A provider can publish a flat carry-forward bar even when nothing traded.
    # Those bars cannot measure the trading range or the overnight move.
    traded = result["volume"].gt(0) & np.isfinite(result["volume"])
    valid_prices = (
        (prices.gt(0) & np.isfinite(prices)).all(axis=1)
        & prices["high"].ge(prices[["open", "close", "low"]].max(axis=1))
        & prices["low"].le(prices[["open", "close"]].min(axis=1))
    )
    valid = traded & valid_prices
    result.loc[~valid, columns] = np.nan
    result["price_available"] = valid.astype(int)
    return result.reset_index()


def collect_prices(start, end, output_dir, all_symbols=False, reuse_dir=None):
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Start must be before end.")
    if pd.Timestamp(end).date() >= pd.Timestamp.now(tz=MARKET_TIMEZONE).date():
        raise ValueError("Download only fully completed dates, ending before today.")
    output = Path(output_dir)
    symbols = list(config.MARKET_SYMBOLS)
    if all_symbols:
        original = pd.read_csv(Path(__file__).with_name("symbols.csv"))["symbol"]
        symbols = list(dict.fromkeys([*original, *symbols]))
    request = {"start": start, "end": end, "symbols": symbols,
               "feed": "sip", "timeframe": "1Day", "adjustment": "split"}
    if all_symbols:
        request["export"] = "ohlc"
    reusable_symbols = set()
    if reuse_dir:
        reuse_dir = Path(reuse_dir)
        previous = json.loads((reuse_dir / "collection_info.json").read_text())["request"]
        for key in ["start", "end", "feed", "timeframe", "adjustment"]:
            if previous[key] != request[key]:
                raise ValueError(f"Cannot reuse raw prices with different {key}.")
        reusable_symbols = set(previous["symbols"])
    manifest_path = output / "collection_info.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["request"] != request:
            raise ValueError("Existing collection settings differ. Use a new output folder.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or matching market-price collection folder.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_daily").mkdir(exist_ok=True)
    client = AlpacaReader(*load_credentials())
    calendar = sorted(client.calendar(start, end), key=lambda row: row["date"])
    if not calendar:
        raise ValueError("No market calendar returned.")
    manifest = {"request": request, "status": "in_progress",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "source": "Alpaca SIP daily stock and ETF bars",
                "availability_policy": "Use at least one TARGET trading session later; never same-day finalized bars.",
                "missing_policy": "Calendar rows retained. OHLC export omits invalid and zero-volume observations.",
                "interpretation": "Exchange-traded share prices. URA holds uranium/nuclear industry companies."}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(calendar).to_csv(output / "market_calendar.csv", index=False)

    def collect_symbol(symbol):
        raw_path = output / "raw_daily" / f"{symbol}.csv"
        if raw_path.exists():
            bars = pd.read_csv(raw_path)
        else:
            previous_raw = reuse_dir / "raw_daily" / raw_path.name if reuse_dir else None
            if symbol in reusable_symbols and previous_raw.is_file():
                bars = pd.read_csv(previous_raw)
            else:
                bars = download_bars(client, symbol, start, end, "sip", mapping_date=end,
                                     timeframe="1Day")
            temporary = raw_path.with_suffix(".partial")
            bars.to_csv(temporary, index=False)
            temporary.replace(raw_path)
        prices = daily_ohlc_on_calendar(bars, calendar) if all_symbols else daily_prices_on_calendar(bars, calendar)
        prices.to_csv(output / f"{symbol}.csv", index=False)
        count = int(prices["price_available"].sum())
        print(f"{symbol}: {count:,}/{len(prices):,} daily prices", flush=True)
        return {"symbol": symbol, "available_prices": count,
                "missing_sessions": len(prices) - count}

    summary = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(collect_symbol, symbol) for symbol in symbols]
        for job in as_completed(jobs):
            summary.append(job.result())
    pd.DataFrame(summary).sort_values("symbol").to_csv(output / "coverage.csv", index=False)
    manifest["status"] = "complete"
    manifest["calendar_sessions"] = len(calendar)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} price histories to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-11-09", help="Keep fixed for the existing experiment")
    parser.add_argument("--output", default="data/market_context_since2016")
    parser.add_argument("--all-symbols", action="store_true", help="Save OHLC for all collected ETFs and major stocks")
    parser.add_argument("--reuse-raw-from", help="Reuse raw daily bars from a compatible Alpaca download")
    args = parser.parse_args()
    try:
        collect_prices(args.start, args.end, args.output, args.all_symbols, args.reuse_raw_from)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Cannot collect market prices: {error}\n")


if __name__ == "__main__":
    main()
