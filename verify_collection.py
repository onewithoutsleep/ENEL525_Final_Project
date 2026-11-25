"""Check a finished collection and make a readable report and daily-data ZIP."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import MINIMUM_EXAMPLES
from data import chronological_split, load_daily_data, make_training_table


def verify_collection(folder):
    folder = Path(folder)
    info = json.loads((folder / "collection_info.json").read_text(encoding="utf-8"))
    if info["status"] != "complete":
        raise ValueError("The collection has not finished successfully for every symbol.")
    summary = pd.read_csv(folder / "collection_summary.csv")
    symbols = pd.read_csv(folder / "symbols.csv")
    calendar = pd.read_csv(folder / "market_calendar.csv", dtype=str)
    if set(summary["symbol"]) != set(symbols["symbol"]) or summary["symbol"].duplicated().any():
        raise ValueError("The collected symbols do not match the requested symbol list.")
    results = []
    known_halt_days = {"2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"}
    for _, item in symbols.iterrows():
        symbol = item["symbol"]
        daily = load_daily_data(folder / "daily" / f"{symbol}.csv")
        quality = pd.read_csv(folder / "quality" / f"{symbol}.csv")
        # A documented market-wide halt is not the same as a provider error.
        # This annotation does not guess prices or change the strict inclusion rule.
        quality["known_march_2020_marketwide_halt_day"] = quality["date"].isin(known_halt_days)
        quality.to_csv(folder / "quality" / f"{symbol}.csv", index=False)
        dates = daily["date"].dt.strftime("%Y-%m-%d").tolist()
        if dates != calendar["date"].tolist() or dates != quality["date"].tolist():
            raise ValueError(f"{symbol}: calendar, daily data, and quality dates differ.")
        table, names = make_training_table(daily)
        source = summary.loc[summary["symbol"] == symbol].iloc[0]
        complete = int(daily["session_complete"].sum())
        if complete != source["complete_sessions"] or len(table) != source["usable_training_examples"]:
            raise ValueError(f"{symbol}: summary counts do not match the saved data.")
        if not np.isfinite(table[names + ["target", "baseline"]].to_numpy()).all():
            raise ValueError(f"{symbol}: invalid values remain in the training examples.")
        # Check the next-session target against the official calendar, including
        # examples next to excluded windows. A missing day must not shift labels.
        following = dict(zip(calendar["date"].iloc[:-1], calendar["date"].iloc[1:]))
        expected_targets = table["forecast_date"].dt.strftime("%Y-%m-%d").map(following)
        if expected_targets.tolist() != table["target_date"].dt.strftime("%Y-%m-%d").tolist():
            raise ValueError(f"{symbol}: a target is not the next official trading session.")
        split_sizes = [0, 0, 0]
        if len(table) >= MINIMUM_EXAMPLES:
            split_sizes = [len(split) for split in chronological_split(table)]
        results.append({
            "symbol": symbol, "type": item["type"], "description": item["description"],
            "complete_sessions": complete, "incomplete_sessions": len(daily) - complete,
            "usable_examples": len(table), "train_rows": split_sizes[0],
            "validation_rows": split_sizes[1], "test_rows": split_sizes[2],
            "enough_for_starter": len(table) >= MINIMUM_EXAMPLES,
        })
    verified = pd.DataFrame(results)
    verified.to_csv(folder / "verified_training_counts.csv", index=False)
    enough = int(verified["enough_for_starter"].sum())
    request = info["request"]
    lines = [
        "# Historical data collection", "",
        f"Collected **{len(verified)} symbols** from Alpaca's **{request['feed'].upper()} feed**,",
        f"requesting **{request['start']} through {request['end']}**.", "",
        f"The market calendar contains **{len(calendar):,} sessions per symbol**.",
        f"The saved raw files contain **{int(summary['raw_bars'].sum()):,} five-minute bars**, including extended hours.",
        f"There are **{int(verified['complete_sessions'].sum()):,} complete regular sessions** across symbols.",
        f"After historical-window and target exclusions, **{int(verified['usable_examples'].sum()):,} labeled examples** remain across symbols.",
        f"**{enough}/{len(verified)} symbols** meet the starter's minimum of {MINIMUM_EXAMPLES} usable examples.", "",
        "These are separate per-symbol datasets, not one pooled training set. Examples across correlated stocks/ETFs are not independent.",
        "No real-data model training or accuracy assessment was performed by this verification step.", "",
        "## Data quality", "",
        "A complete session contains every expected five-minute regular-session bar, with valid prices and volume.",
        "Incomplete sessions remain as dated rows with blanks and session_complete=0. No returns are invented.",
        "The training code excludes affected history windows and targets. This strict rule can exclude many examples after a small number of scattered gaps.", "",
        "**Stress-period limitation:** flagged dates include March 9, 12, 16, and 18, 2020, when documented market-wide circuit breakers halted trading. The quality files identify those dates. Missing bars during a halt are not necessarily a data-provider error. The strict starter policy excludes affected windows, so it should not be used to claim performance on those crisis sessions without a separately justified halt-handling method. Original bars are preserved.",
        "[NYSE's account of the March 2020 halts](https://www.nyse.com/article/assessing-nyse-model-performance)", "",
        "All daily and quality files were checked against the official session calendar. Usable inputs are finite; targets refer to the next official session.",
        "Training/validation/test counts below use the default 20-day window and approximately 60/20/20 split, with boundary gaps.", "",
        "| Symbol | Type | Complete days | Incomplete days | Usable examples | Train | Validation | Test |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(f"| {row['symbol']} | {row['type']} | {row['complete_sessions']} | {row['incomplete_sessions']} | {row['usable_examples']} | {row['train_rows']} | {row['validation_rows']} | {row['test_rows']} |")
    lines += ["", "## Using the data", "",
              "Each daily/SYMBOL.csv file can be passed to train.py. See COLLECT_DATA.md in the project folder for commands and definitions.",
              "Raw data is cached by symbol and year under raw_5minute/. The daily-data ZIP includes all daily files, quality reports, and provenance, but not the much larger raw files.", "",
              "The target is regular-session realized volatility from five-minute log returns, including the first bar's open-to-close return and excluding overnight movement. Values are decimals, not annualized.",
              "Prices/volumes are split-adjusted. The symbol sample was selected today and has survivorship/selection bias; it is not a point-in-time historical stock universe.", "",
              "[Alpaca historical bar documentation](https://docs.alpaca.markets/us/reference/stockbars)"]
    (folder / "COLLECTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = folder.parent / "ETF_Stock_Daily_Data_2016_2026.zip"
    files = [file for name in ["daily", "quality"] for file in (folder / name).glob("*.csv")]
    files += [file for file in folder.iterdir() if file.is_file()]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for file in files:
            bundle.write(file, file.relative_to(folder))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("The daily-data archive failed its integrity check.")
    print(f"Verified {len(verified)} symbols; {enough} meet the training minimum.")
    print(f"Complete sessions: {int(verified.complete_sessions.sum()):,}")
    print(f"Usable examples: {int(verified.usable_examples.sum()):,}")
    print(f"Daily-data ZIP: {archive.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="data/alpaca_since2016")
    args = parser.parse_args()
    verify_collection(args.folder)
