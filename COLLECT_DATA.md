# Collect history since 2016 for 40 ETFs and stocks

`collect_alpaca.py` downloads the 40 tickers in `symbols.csv` and converts their
five-minute bars into the daily CSV format used by the models. The default
dates are **January 1, 2016 through November 9, 2026**, inclusive. The first
expected trading session is January 4, 2016. This expands the original proposal's
3-5 years at your request. Available/complete history is reported per symbol.

The selection contains 20 ETFs covering US and international stocks, sectors,
bonds, gold, and silver; and 20 major US-listed stocks. It is a hand-picked
research sample of current symbols, not a point-in-time historical universe.
It therefore has selection/survivorship bias and does not represent every
company that existed at the start of the sample. Several ETFs overlap in their holdings.

## Connect your account privately

Signing in on the Alpaca website does not give Python API access. Open Alpaca,
switch to Paper Trading, and locate API Keys. Use your existing paper key ID
and secret if you have them, or create paper API credentials on the website.
If the website only offers regeneration, remember that regenerating replaces
the old credentials used by other programs.

Run the local setup window from this project folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\connect_alpaca.ps1
```

Enter both keys in that window, not in a chat, then choose **Save keys locally**.
Windows encrypts both values in `.alpaca-credentials.xml` for your Windows user.
This file is excluded from version control and the project download archives.
It is stored in this project folder, so an encrypted copy may sync with the
folder; another Windows user/computer cannot normally decrypt that copy.

Alternatively, the collector reads `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`
environment variables. It never prints credentials, sends them to another
service, submits orders, or changes the account. Only Alpaca's data and market
calendar endpoints are called.

## Download

After installing the main project's requirements, run:

```powershell
.\.venv\Scripts\python.exe collect_alpaca.py
```

The default is Alpaca's **SIP** consolidated feed. Alpaca documents access to
older historical SIP data without the real-time subscription when the end time
is at least 15 minutes old. The collector uses completed prior sessions and
does not buy or upgrade a subscription. If your account denies access, it stops
or reports the failed symbols; it does not silently switch to a different feed.

The alternative `--feed iex` uses only the IEX exchange, so its volume and
volatility measure are different. Use a new output directory if changing feeds:

```powershell
.\.venv\Scripts\python.exe collect_alpaca.py --feed iex --output data/alpaca_iex_since2016
```

This is a multi-million-bar download. It follows all response pages, retries
temporary failures, and limits request frequency. If interrupted, run the same
command again: completed symbol/year downloads are reused. The interrupted
year restarts. A different date range or feed requires a new output folder.
Three symbols download concurrently by default, with a shared rate limit;
use `--workers 1` to download sequentially.
If large requests repeatedly time out, rerun with `--page-size 1000` to ask for
smaller response pages. The saved data is reused; this does not change the
requested dates, feed, or volatility calculation.

## Files produced

Everything is saved under `data/alpaca_since2016` by default:

| File/folder | Contents |
| --- | --- |
| `raw_5minute/SYMBOL/YEAR.csv.gz` | Compressed five-minute data returned by Alpaca, including any extended-hours bars. Prices and volumes are split-adjusted. |
| `daily/SYMBOL.csv` | One row for every expected trading session, directly readable by the training code. |
| `quality/SYMBOL.csv` | Expected/received intervals, missing bars, duplicates, and completeness flags. |
| `market_calendar.csv` | Alpaca's session dates and opening/closing times, including early closes. |
| `collection_summary.csv` | Download status, counts, coverage, and usable training examples per symbol. |
| `collection_info.json` | Source, dates, feed, adjustment, and calculation definitions. |
| `symbols.csv` | The exact requested symbol list. |

Each instrument stays separate. Do not combine different stocks into one daily
file or treat the last day of one stock as the day before another stock.

## How daily values are calculated

Only the exchange's **regular session**, interpreted in `America/New_York`,
contributes to daily values. Daylight-saving time and official early closes are
handled using the calendar. An ordinary full session has 78 five-minute bars;
a session ending at 13:00 has 42. These are checked by exact timestamps, not
just by counting bars.

For each complete session:

```text
first return = log(first 5-minute close / first 5-minute open)
later return = log(current close / previous 5-minute close)
realized_volatility = sqrt(sum(returns squared))
close = last regular-session bar's close
volume = sum(regular-session bar volumes)
```

Volatility is a decimal, not annualized. Overnight returns are excluded from
this target. The first bar's open is the first eligible trade in that bar; it
is not necessarily the official opening-auction price. Stock splits are
adjusted; cash dividends are not. This is a currently adjusted historical
snapshot, not a reconstruction of the data as originally seen on each date.

If any required bar is missing, duplicated, misaligned, or invalid, the session
is retained with `session_complete=0` and blank numeric values. Missing bars
can reflect halts, no eligible trades, or data gaps; the collector does not
invent returns or assume the cause. The training loader preserves that date
and excludes any input/target window affected by it. It never shifts the
following available session backward across the gap. This strict policy can
substantially reduce the usable sample; check the summary and quality reports.

The flags validate timestamp coverage and basic price/volume consistency, not
every possible error in the data source. No extreme market moves are removed
automatically. Source data may be revised after retrieval.

In the collected sample, several flagged dates coincide with the documented
market-wide halts on March 9, 12, 16, and 18, 2020. The final verification adds
an identifying column to each quality report. Under the strict starter rule,
these sessions and affected historical windows are excluded, which limits
claims about crisis-period performance. A later study should explicitly choose
and document a halt-handling method before evaluating those days; the original
bars remain available. [NYSE's event summary](https://www.nyse.com/article/assessing-nyse-model-performance).

## Train one symbol

On this computer, the daily files are already saved inside the project. If
moving the project to another computer, extract the daily-data ZIP **inside
`volatility_project/data/alpaca_since2016`**, so that folder contains `daily`,
`quality`, and the collection report. Then the command below uses the right path.

```powershell
.\.venv\Scripts\python.exe train.py --data data/alpaca_since2016/daily/SPY.csv --output results_SPY_01
```

Each ticker needs at least 100 usable examples after all windows and quality
exclusions. The current models train one ticker at a time. Pooling tickers into
one model would require additional code and a shared date-based split to avoid
training on one stock's future while testing on another stock's past.

## References

- [Alpaca historical bars and pagination](https://docs.alpaca.markets/us/reference/stockbars)
- [Alpaca historical SIP access and feed differences](https://docs.alpaca.markets/us/docs/market-data-faq)
- [Alpaca market calendar](https://docs.alpaca.markets/us/v1.1/reference/getcalendar-1)

Offline checks: `python -m unittest -v test_collection test_workflow`.

After collection completes, run `python verify_collection.py` to check every
symbol's dates and usable training rows and produce a collection report and a
ZIP containing the daily data, quality reports, and source details.
