"""Turn saved validation predictions into a readable table, chart, and report."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "neural_network": "Neural network",
}


def make_report(results_dir, review_dir, symbol):
    results = Path(results_dir)
    output = Path(review_dir)
    info = json.loads((results / "training_info.json").read_text(encoding="utf-8"))
    review = json.loads((output / "training_review.json").read_text(encoding="utf-8"))
    if info.get("test_evaluated") is not False or (results / "test_predictions.csv").exists():
        raise ValueError("This pilot report requires a validation-only run.")
    if review["last_session"] != info["training_last_target_date"]:
        raise ValueError("The training review and model run use different date ranges.")
    if (review["input_count"] != len(info["feature_names"])
            or Path(review["data_file"]).resolve() != Path(info["data_file"]).resolve()):
        raise ValueError("The training review and model run must use the same data and inputs.")
    scores = pd.read_csv(results / "scores.csv")
    if set(scores["split"]) != {"validation"}:
        raise ValueError("Only validation scores may be used in this report.")
    prediction = pd.read_csv(
        results / "validation_predictions.csv", parse_dates=["forecast_date", "target_date"]
    )
    baseline = info["baseline_name"]
    labels = {baseline: f"{info['baseline_days']}-day average", **MODEL_LABELS}
    comparison = scores.sort_values("QLIKE").copy()
    comparison["Model"] = comparison["model"].map(labels)
    comparison["MAE_percentage_points"] = comparison["MAE"] * 100
    comparison = comparison[["Model", "MAE_percentage_points", "MSE", "QLIKE",
                             "QLIKE_improvement_percent", "meets_5_percent_goal"]]
    comparison.to_csv(output / "model_comparison.csv", index=False)

    # Each panel shows the same dates and units, with one model to avoid clutter.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)
    colors = ["#2366ad", "#167568", "#8b4aa8"]
    dates = prediction["target_date"]
    for ax, (name, label), color in zip(axes, MODEL_LABELS.items(), colors):
        ax.plot(dates, prediction["target"] * 100, label="Actual", color="#37424e",
                linewidth=0.85, alpha=0.8)
        ax.plot(dates, prediction["baseline"] * 100, label=f"{info['baseline_days']}-day average",
                color="#c68a16", linewidth=1.15, linestyle="--")
        ax.plot(dates, prediction[name] * 100, label=label, color=color, linewidth=1.15)
        score = scores.loc[scores["model"] == name].iloc[0]
        ax.set_title(f"{label}  |  QLIKE {score['QLIKE']:.4f}", loc="left", fontsize=12, weight="bold")
        ax.set_ylabel("Daily volatility (%)")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=9)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlabel("Session being predicted")
    fig.suptitle(f"{symbol} | Next-day volatility on validation dates", x=0.08, ha="left",
                 fontsize=20, weight="bold")
    fig.text(0.08, 0.94,
             f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}  |  "
             f"{len(prediction):,} forecasts  |  {info['lookback_days']}-day history, "
             f"{len(info['feature_names'])} inputs", color="#46505b")
    fig.text(0.08, 0.025,
             "Every forecast uses information available after the previous session. Lower QLIKE is better.\n"
             "Regular-session realized volatility; no annualization. The final test period has not been scored.",
             fontsize=10, color="#46505b")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.11, hspace=0.32)
    fig.savefig(output / "validation_predictions.png", dpi=170)
    fig.savefig(output / "validation_predictions.svg")
    plt.close(fig)

    # Inspect training diagnostics from our own saved model, never test answers.
    bundle = joblib.load(results / "models.joblib")
    network = bundle["models"]["neural_network"].regressor_.named_steps["network"]
    diagnostics = {
        "network_training_passes": int(network.n_iter_),
        "network_final_training_loss": float(network.loss_),
        "network_training_loss_history": [float(value) for value in network.loss_curve_],
        "warnings": info["convergence_notes"],
        "matplotlib_version": matplotlib.__version__,
    }
    (output / "training_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    # Fingerprints let a later experiment detect changed data or source code.
    project = Path(__file__).resolve().parent
    paths = [Path(info["data_file"])] + [project / name for name in
             ["config.py", "data.py", "models.py", "metrics.py", "train.py"]]
    fingerprints = {}
    for path in paths:
        with path.open("rb") as stream:
            fingerprints[str(path.resolve())] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "input_fingerprints.json").write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")

    winner = info["selected_model_by_validation"]
    winner_row = scores.loc[scores["model"] == winner].iloc[0]
    ranges = info["date_ranges"]
    flags = pd.read_csv(output / "training_incomplete_sessions.csv")
    flagged_dates = ", ".join(flags["date"].astype(str)) or "none"
    halt_column = "known_march_2020_marketwide_halt_day"
    halt_dates = ", ".join(flags.loc[flags[halt_column].eq(True), "date"].astype(str)) if halt_column in flags else ""
    experiment_description = (
        "This run used the parameter search listed in config.py; all settings were selected on validation. "
        if info.get("tuning_enabled") else
        "This is a first comparison using the original settings: one configuration per model, with no parameter search. "
    )
    lines = [
        f"# {symbol}: first real-data volatility experiment",
        "",
        f"**{labels[winner]} had the lowest validation QLIKE ({winner_row['QLIKE']:.4f}).** "
        f"The final {ranges['test']['rows']:,} test examples remain unscored.",
        "",
        experiment_description + "These development results do not establish final test performance.",
        "",
        "## What the models predict",
        "",
        f"After each close, predict the following trading session's realized volatility. "
        f"All models use {info['lookback_days']} trading days of history and "
        f"{len(info['feature_names'])} inputs. The baseline averages the previous "
        f"{info['baseline_days']} observed daily volatilities. These are forecasts of movement size, not direction.",
        "",
        "## Date ranges",
        "",
        "| Portion | Examples | Forecast dates | Target dates | Use |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for name, purpose in [("train", "Fit models and scaler"), ("validation", "Compare models"),
                          ("test", "Reserved; no predictions or scores")]:
        span = ranges[name]
        lines.append(f"| {name.title()} | {span['rows']:,} | "
                     f"{span['first_forecast_date']} to {span['last_forecast_date']} | "
                     f"{span['first_target_date']} to {span['last_target_date']} | {purpose} |")
    lines += [
        "", "One example is omitted at each split boundary. Models and input scaling were fitted "
        "only on training rows. Later observed history is used as it becomes available for each one-day forecast.",
        "", "## Validation comparison", "",
        "| Model | MAE (percentage points) | MSE (decimal volatility squared) | QLIKE | QLIKE improvement vs. baseline |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.iterrows():
        lines.append(f"| {row['Model']} | {row['MAE_percentage_points']:.4f} | {row['MSE']:.8f} | "
                     f"{row['QLIKE']:.4f} | {row['QLIKE_improvement_percent']:+.2f}% |")
    lines += ["", "Relative to the 20-day average baseline:", ""]
    for _, row in scores.loc[scores["model"] != baseline].iterrows():
        improvement = row["QLIKE_improvement_percent"]
        direction = "lower" if improvement >= 0 else "higher"
        lines.append(f"- {labels[row['model']]}: {abs(improvement):.1f}% {direction} validation QLIKE error.")
    lines += [
        "", "Lower MAE, MSE, and QLIKE are better. Positive improvement means lower QLIKE than the baseline. "
        "MAE of 0.10 percentage points means an average absolute error of 0.001 in decimal daily volatility.",
        "", "QLIKE compares squared volatilities: mean(actual_variance / predicted_variance "
        "- log(actual_variance / predicted_variance) - 1), with a small numerical floor. "
        "It penalizes severe underprediction strongly. Model rankings can differ between error measures.",
        "", "The 5% QLIKE target is a validation screening criterion here. "
        "It still needs an untouched final test and uncertainty analysis before a project success claim.",
        "", "![Validation predictions](validation_predictions.png)",
        "", "## Training-data review and exclusions", "",
        f"Reviewed {review['calendar_sessions']:,} training-period calendar sessions from "
        f"{review['first_session']} through {review['last_session']}: "
        f"{review['complete_sessions']:,} complete and {review['incomplete_sessions']} incomplete. "
        f"After the history/next-day requirement, incomplete sessions remove "
        f"{review['examples_lost_to_incomplete_sessions']} additional training examples, "
        f"leaving {review['training_examples']:,} usable examples.",
        "", "The strict completeness policy is unchanged. Incomplete dates stay in sequence with missing "
        "values, and affected inputs and targets are excluded. No prices, returns, or labels are filled. "
        "Large valid observations are retained; there is no outlier clipping.",
        "", f"Flagged training dates: {flagged_dates}. "
        + (f"Dates identified in the quality file as market-wide halt days: {halt_dates}. " if halt_dates else "")
        + "Absent bars during a halt are not automatically provider errors. "
        "This pilot cannot demonstrate performance on excluded crisis sessions. "
        "A separate halt-aware aggregation experiment is needed before drawing crisis-period conclusions.",
        "", "![Training-period review](training_review.png)",
        "", "Data uses split-adjusted Alpaca SIP five-minute bars during regular sessions. "
        "Daily volatility is the square root of summed intraday squared log returns, including the first "
        "five-minute open-to-close return and excluding overnight returns. It is not annualized. "
        "The last bar close is used as the daily close; prices are not dividend-adjusted.",
        "", "## Settings and training checks", "",
        "```json", json.dumps(info["chosen_settings"], indent=2), "```", "",
        f"Random seed: {info['random_seed']}. Python: {info['python_version']}. "
        f"scikit-learn: {info['sklearn_version']}. The network completed {network.n_iter_} training passes. "
        "Its scaler learned only from training inputs. Its training loss is squared log-volatility error; "
        "the trees fit squared volatility error. QLIKE chooses the preferred model.",
        "", "Training warnings: " + ("; ".join(info["convergence_notes"]) or "none recorded."),
        "", "## Next experiment", "",
        "Keep this result as the reference. Try the small predefined parameter search on validation only, "
        "then consider a simpler network or more scale-independent inputs if its validation error remains high. "
        "Compare every change with the same baseline and dates. Define halt handling and a walk-forward "
        "validation plan before the final test.",
        "", "The current split is based on usable-row fractions. Before changing history length, exclusion "
        "rules, or the data end date, freeze these calendar boundaries in the splitting code; otherwise "
        "the held-out date ranges can move. Do not treat different symbols' overlapping dates as independent trials.",
        "", "## Files", "",
        "- `model_comparison.csv`: the comparison table, suitable for Excel.",
        "- `validation_predictions.png` / `.svg`: prediction chart (SVG stays sharp when resized).",
        "- `training_review.png`, `training_summary.csv`, `training_incomplete_sessions.csv`: training data review.",
        "- `models/validation_predictions.csv`: all individual validation forecasts and actual values.",
        "- `models/models.joblib`: the three fitted models, ready for `predict.py`.",
        "- `models/training_info.json`: date boundaries, settings, and input definitions.",
        "- `training_diagnostics.json`, `input_fingerprints.json`: training history and reproducibility checks.",
        "", "No test predictions or test scores are included.", "",
    ]
    (output / f"{symbol}_PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"\nSaved report and charts to: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Folder saved by a validation-only train.py run")
    parser.add_argument("--review", required=True, help="Folder saved by review_training_data.py")
    parser.add_argument("--symbol", required=True)
    args = parser.parse_args()
    make_report(args.results, args.review, args.symbol)


if __name__ == "__main__":
    main()
