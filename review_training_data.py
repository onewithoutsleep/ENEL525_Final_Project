"""Review prices, volume, and volatility in the TRAINING period only."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts to files; no desktop window is needed.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

import config
from data import chronological_split, load_daily_data, make_training_table


def review_training_data(data_path, quality_path, output_dir, symbol):
    daily = load_daily_data(data_path)
    table, names = make_training_table(daily)
    train, _, _ = chronological_split(
        table, config.TRAIN_FRACTION, config.VALIDATION_FRACTION
    )
    # Include the final training answer, but no validation or test prices/answers.
    cutoff = train["target_date"].max()
    history = daily.loc[daily["date"] <= cutoff].copy()
    quality = pd.read_csv(quality_path, parse_dates=["date"])
    quality = quality.loc[quality["date"] <= cutoff].copy()
    if not quality["date"].reset_index(drop=True).equals(history["date"].reset_index(drop=True)):
        raise ValueError("The daily and quality files do not have the same training dates.")
    incomplete = history.loc[history["session_complete"] == 0, "date"]
    flags = quality.loc[quality["date"].isin(incomplete)].copy()

    # Compare available examples with the count if every session were complete.
    warmup = max(config.LOOKBACK_DAYS, config.BASELINE_DAYS - 1, 4)
    potential = max(0, len(history) - warmup - 1)
    summary = {
        "symbol": symbol,
        "data_file": str(Path(data_path).resolve()),
        "first_session": str(history["date"].min().date()),
        "last_session": str(cutoff.date()),
        "calendar_sessions": len(history),
        "complete_sessions": int((history["session_complete"] == 1).sum()),
        "incomplete_sessions": len(flags),
        "lookback_days": config.LOOKBACK_DAYS,
        "input_count": len(names),
        "training_examples": len(train),
        "potential_examples_with_complete_data": potential,
        "examples_lost_to_incomplete_sessions": potential - len(train),
        "policy": "Keep incomplete dates; exclude affected inputs/targets after alignment. No filling.",
        "validation_or_test_values_reviewed": False,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "training_review.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flags.to_csv(output / "training_incomplete_sessions.csv", index=False)
    history[["close", "volume", "realized_volatility"]].describe(
        percentiles=[0.01, 0.5, 0.99]
    ).to_csv(output / "training_summary.csv")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    series = [
        ("close", 1, "Close (USD)", "#2563a6"),
        ("volume", 1e-6, "Volume (million shares)", "#47836c"),
        ("realized_volatility", 100, "Daily realized volatility (%)", "#7650a8"),
    ]
    for ax, (column, scale, label, color) in zip(axes, series):
        ax.plot(history["date"], history[column] * scale, color=color, linewidth=0.85)
        for number, date in enumerate(incomplete):
            ax.axvline(date, color="#bc403c", alpha=0.45, linewidth=0.9,
                       label="Incomplete session" if number == 0 else None)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="upper left", frameon=False)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Trading session")
    fig.suptitle(f"{symbol} | Training-period data review", x=0.09, ha="left", fontsize=19, weight="bold")
    fig.text(0.09, 0.935,
             f"{summary['first_session']} to {summary['last_session']}  |  "
             f"{len(history):,} sessions  |  {len(flags)} incomplete", color="#46505b")
    fig.text(0.09, 0.025,
             "Incomplete sessions remain gaps. March 2020 halt dates and affected history windows are excluded.\n"
             "Volatility uses regular-session five-minute returns; overnight returns are excluded. No annualization.",
             fontsize=9, color="#46505b")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.12, hspace=0.18)
    fig.savefig(output / "training_review.png", dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print("\nIncomplete training sessions:")
    print(flags.to_string(index=False))
    print("\nTraining-period descriptive statistics:")
    print(history[["close", "volume", "realized_volatility"]].describe().to_string())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--output", required=True, help="A NEW report folder")
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    review_training_data(args.data, args.quality, args.output, args.symbol)


if __name__ == "__main__":
    main()
