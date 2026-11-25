"""Offline checks for the collector; no credentials or network calls required."""

import tempfile
import unittest
import gzip
import http.client
import io
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from collect_alpaca import aggregate_daily, download_bars
from data import load_daily_data, make_training_table
from make_demo_data import make_demo_data
from alpaca_access import AlpacaReader


def fake_session(date, close="16:00", price=100.0):
    timestamps = pd.date_range(f"{date} 09:30", f"{date} {close}",
                               tz="America/New_York", freq="5min", inclusive="left")
    # One 1% log move in the first interval, then no further movement.
    last_price = price * np.exp(0.01)
    return pd.DataFrame({
        "timestamp": timestamps.tz_convert("UTC"), "open": price,
        "high": last_price, "low": price, "close": last_price, "volume": 1000,
    })


class CollectionTests(unittest.TestCase):
    def test_connection_drop_is_retried_and_gzip_is_read(self):
        response = io.BytesIO(gzip.compress(b'{"ok": true}'))
        response.headers = {"Content-Encoding": "gzip"}
        reader = AlpacaReader("test-key-not-real", "test-secret-not-real")
        with patch("alpaca_access.time.sleep"), patch(
            "alpaca_access.urllib.request.urlopen",
            side_effect=[http.client.RemoteDisconnected("test disconnect"), response],
        ) as request:
            result = reader.get("data.alpaca.markets", "/v2/stocks/SPY/bars", {})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(request.call_count, 2)

    def test_early_close_first_return_and_no_overnight_return(self):
        calendar = [{"date": "2024-11-27", "open": "09:30", "close": "16:00"},
                    {"date": "2024-11-29", "open": "09:30", "close": "13:00"}]
        bars = pd.concat([fake_session("2024-11-27"), fake_session("2024-11-29", "13:00", 200)])
        daily, audit = aggregate_daily(bars, calendar)
        self.assertEqual(audit["expected_bars"].tolist(), [78, 42])
        self.assertEqual(daily["session_complete"].tolist(), [1, 1])
        np.testing.assert_allclose(daily["realized_volatility"], [0.01, 0.01])
        self.assertEqual(daily["volume"].tolist(), [78000, 42000])

    def test_missing_and_duplicate_bars_are_flagged(self):
        calendar = [{"date": "2024-11-27", "open": "09:30", "close": "16:00"}]
        bars = fake_session("2024-11-27")
        misaligned = bars.copy()
        misaligned.loc[10, "timestamp"] += pd.Timedelta(minutes=1)
        for bad in [bars.iloc[1:], pd.concat([bars, bars.iloc[[0]]]), bars.iloc[:0], misaligned]:
            daily, audit = aggregate_daily(bad, calendar)
            self.assertEqual(daily["session_complete"].iloc[0], 0)
            self.assertTrue(pd.isna(daily["realized_volatility"].iloc[0]))
            self.assertFalse(audit["session_complete"].iloc[0])

    def test_daylight_saving_and_extended_hours(self):
        calendar = [{"date": date, "open": "09:30", "close": "16:00"}
                    for date in ["2024-03-08", "2024-03-11"]]
        first, second = fake_session("2024-03-08"), fake_session("2024-03-11")
        self.assertEqual(first["timestamp"].iloc[0].hour, 14)
        self.assertEqual(second["timestamp"].iloc[0].hour, 13)
        extra = first.iloc[[0]].copy()
        extra["timestamp"] = extra["timestamp"] - pd.Timedelta(hours=1)
        extra["volume"] = 999999
        daily, _ = aggregate_daily(pd.concat([first, second, extra]), calendar)
        self.assertEqual(daily["volume"].tolist(), [78000, 78000])
        self.assertEqual(daily["session_complete"].tolist(), [1, 1])

    def test_incomplete_session_does_not_join_separate_days(self):
        daily = make_demo_data(200)
        daily["session_complete"] = 1
        daily.loc[80, "session_complete"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            daily.to_csv(path, index=False)
            loaded = load_daily_data(path)
            self.assertEqual(len(loaded), 200)
            self.assertTrue(pd.isna(loaded.loc[80, "close"]))
            table, _ = make_training_table(loaded)
            # t=79 would target the invalid session; t=80..100 use it in their
            # input histories. Their removal must not change other target dates.
            invalid_dates = pd.to_datetime(daily.loc[79:100, "date"])
            self.assertFalse(table["forecast_date"].isin(invalid_dates).any())
            first_after = table[table["forecast_date"] > invalid_dates.max()].iloc[0]
            self.assertEqual(first_after["forecast_date"], pd.Timestamp(daily.loc[101, "date"]))
            self.assertEqual(first_after["target_date"], pd.Timestamp(daily.loc[102, "date"]))

    def test_all_pages_are_requested(self):
        class FakeReader:
            def __init__(self):
                self.calls = []

            def get(self, host, path, parameters):
                self.calls.append(dict(parameters))
                index = len(self.calls)
                return {"bars": [{"t": f"2024-01-02T14:{25 + index * 5}:00Z",
                                  "o": 100, "h": 101, "l": 99, "c": 100, "v": 10}],
                        "next_page_token": "page2" if index == 1 else None}

        reader = FakeReader()
        bars = download_bars(reader, "SPY", "2024-01-02", "2024-01-02", "sip", page_size=1000)
        self.assertEqual(len(bars), 2)
        self.assertEqual(reader.calls[1]["page_token"], "page2")
        self.assertEqual(reader.calls[0]["adjustment"], "split")
        self.assertEqual(reader.calls[0]["limit"], 1000)


if __name__ == "__main__":
    unittest.main()
