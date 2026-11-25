"""Create FAKE daily data to check the code. Never use it as project evidence."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_demo_data(days=500, seed=42):
    if days < 1:
        raise ValueError("The number of days must be positive.")
    rng = np.random.default_rng(seed)
    volatility = np.empty(days)
    volatility[0] = 0.015
    for day in range(1, days):
        # Invent persistent volatility, not a realistic market simulator.
        volatility[day] = np.exp(
            0.93 * np.log(volatility[day - 1])
            + 0.07 * np.log(0.015) + rng.normal(0, 0.12)
        )
    returns = rng.normal(0, volatility)
    return pd.DataFrame({
        # Business days are only labels here; they are NOT an exchange calendar.
        "date": pd.bdate_range("2020-01-01", periods=days).strftime("%Y-%m-%d"),
        "close": 100 * np.exp(np.cumsum(returns)),
        "volume": rng.lognormal(14, 0.25, days).astype(int),
        "realized_volatility": volatility,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=500)
    parser.add_argument("--output", default="data/SYNTHETIC_daily.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.exit(1, f"File already exists: {output}. Choose a different --output name.\n")
    make_demo_data(args.days).to_csv(output, index=False)
    print(f"Created {output}: SYNTHETIC data for checking the workflow only.")


if __name__ == "__main__":
    main()
