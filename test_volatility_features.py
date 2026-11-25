"""Check the estimator against its formula and verify input timing and gaps."""

import unittest

import numpy as np
import pandas as pd

from collect_market_prices import daily_ohlc_on_calendar
from data import make_training_table
from make_demo_data import make_demo_data
from volatility_features import (
    build_ohlc_features, daily_variance_terms, history_rows_needed, yang_zhang_variance,
)


class VolatilityTests(unittest.TestCase):
    def make_prices(self, dates):
        close = 100 + np.arange(len(dates)) * 0.2
        opening = close + np.sin(np.arange(len(dates)))
        return pd.DataFrame({
            "open": opening, "high": np.maximum(opening, close) + 2,
            "low": np.minimum(opening, close) - 1, "close": close,
        }, index=pd.DatetimeIndex(dates))

    def test_formula_uses_prior_close_and_sample_variances(self):
        bars = self.make_prices(pd.date_range("2024-01-01", periods=5))
        overnight, intraday, range_var = daily_variance_terms(bars)
        o, h, l, c = bars.to_numpy().T
        expected_gap = np.log(o[1:] / c[:-1])
        expected_intraday = np.log(c[1:] / o[1:])
        expected_range = np.log(h[1:] / o[1:]) * np.log(h[1:] / c[1:])
        expected_range += np.log(l[1:] / o[1:]) * np.log(l[1:] / c[1:])
        k = 0.34 / (1.34 + 5 / 3)
        expected = np.var(expected_gap, ddof=1) + k * np.var(expected_intraday, ddof=1)
        expected += (1 - k) * expected_range.mean()
        result = yang_zhang_variance(overnight, intraday, range_var, 4)
        self.assertAlmostEqual(result.iloc[-1], expected, places=14)
        self.assertTrue(result.iloc[:-1].isna().all())
        self.assertAlmostEqual(overnight.iloc[1], expected_gap[0], places=14)
        with self.assertRaises(ValueError):
            yang_zhang_variance(overnight, intraday, range_var, 1)

    def test_missing_day_never_becomes_an_overnight_return(self):
        bars = self.make_prices(pd.date_range("2024-01-01", periods=22))
        bars.iloc[10] = np.nan
        terms = daily_variance_terms(bars)
        self.assertTrue(terms[0].iloc[10:12].isna().all())
        self.assertTrue(pd.isna(yang_zhang_variance(*terms, 21).iloc[-1]))
        self.assertTrue(np.isfinite(yang_zhang_variance(*terms, 21, 0.9).iloc[-1]))
        self.assertTrue(pd.isna(yang_zhang_variance(*terms, 10, 0.9).iloc[15]))

    def test_daily_slots_totals_and_forecast_cutoff(self):
        dates = pd.date_range("2024-01-01", periods=40)
        daily = pd.DataFrame({"date": dates})
        bars = self.make_prices(dates)
        options = ("SPY", 20, 1, 10, (21,), 0.9)
        before = build_ohlc_features(daily, {"SPY": bars}, *options)
        changed = bars.copy()
        changed.iloc[30:] *= 3
        after = build_ohlc_features(daily, {"SPY": changed}, *options)
        pd.testing.assert_frame_equal(before.iloc[:31], after.iloc[:31])
        overnight, intraday, range_var = daily_variance_terms(bars)
        self.assertAlmostEqual(before.loc[30, "market_SPY_daily_volatility_lag_10"],
                               np.sqrt(overnight.iloc[20] ** 2 + range_var.iloc[20]))
        variance = yang_zhang_variance(overnight, intraday, range_var, 21, 0.9)
        self.assertAlmostEqual(before.loc[30, "market_SPY_total_yang_zhang_21d"],
                               np.sqrt(21 * variance.iloc[29]))

    def test_year_window_warmup_preserves_next_session_target(self):
        daily = make_demo_data(400)
        daily["date"] = pd.to_datetime(daily["date"])
        bars = self.make_prices(daily["date"])
        context = {
            "target_symbol": "SPY", "lookback_days": 20, "delay_sessions": 1,
            "volatility_days": 10, "horizons": (21, 126, 252), "minimum_coverage": 0.9,
        }
        table, names = make_training_table(daily, ohlc_prices={"SPY": bars}, ohlc_context=context)
        self.assertEqual(history_rows_needed(context), 254)
        self.assertEqual(table.iloc[0]["forecast_date"], daily["date"].iloc[253])
        self.assertEqual(table.iloc[0]["target"], daily["realized_volatility"].iloc[254])
        self.assertEqual(table.iloc[0]["target_date"], daily["date"].iloc[254])
        self.assertEqual(len(names), 44 + 17)

    def test_zero_volume_flat_bar_is_not_a_volatility_observation(self):
        bars = pd.DataFrame({
            "timestamp": ["2024-03-08T05:00:00Z", "2024-03-11T04:00:00Z"],
            "open": [100, 100], "high": [101, 100], "low": [99, 100],
            "close": [100, 100], "volume": [500, 0],
        })
        calendar = [{"date": date} for date in ["2024-03-08", "2024-03-11"]]
        result = daily_ohlc_on_calendar(bars, calendar)
        self.assertEqual(result["price_available"].tolist(), [1, 0])
        self.assertTrue(pd.isna(result["close"].iloc[1]))


if __name__ == "__main__":
    unittest.main()
