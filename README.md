# ENEL 525: next-day volatility models

This Python starter project contains a **decision tree**, a **random forest**
(the tree-based model in your proposal), and a **feedforward neural network**.
The first real-data SPY experiment is in
**[reports/SPY_pilot_01/SPY_PILOT_REPORT.md](reports/SPY_pilot_01/SPY_PILOT_REPORT.md)**.
It compares all three models using 20 days of history and keeps the final test
period unscored. The synthetic demo is also available for practicing the workflow.

To download five-minute history for 40 ETFs and stocks from your Alpaca account,
start with **[COLLECT_DATA.md](COLLECT_DATA.md)**. It explains the private local
connection window, collection commands, daily conversion, and quality reports.

**The network uses 20 trading days of information for every prediction.**
The tree and forest receive the same inputs for a fair comparison.
Predictions are made **after today's trading session has finished**, for the
next trading session. These are volatility forecasts, not stock-price forecasts.

## 1. Where to start reading

| File | Purpose |
| --- | --- |
| `models.py` | Creates the decision tree, forest, and neural network. |
| `config.py` | Settings: history length, tree sizes, network layers, learning rate. |
| `data.py` | Reads your CSV, builds historical inputs, and splits by date. |
| `train.py` | Trains, compares, and saves all three models. |
| `predict.py` | Loads saved models and makes the next one-day forecast. |
| `metrics.py` | Calculates MAE, MSE, and QLIKE. |
| `make_demo_data.py` | Creates clearly labeled fake data for trying the workflow. |
| `data/daily_template.csv` | Empty template for your real daily data. |
| `test_workflow.py` | Checks timing, input preparation, metrics, and saved models. |
| `review_training_data.py` | Reviews training-period prices, volume, volatility, and incomplete sessions. |
| `make_validation_report.py` | Creates a comparison table and chart from validation results. |

The models use scikit-learn, so you do not need to write backpropagation or tree
splitting from scratch. Comments explain the main steps.

## 2. Install Python and the packages

Use Python **3.12 or newer**. If `python --version` does not work, install Python
from [python.org](https://www.python.org/downloads/) and enable its PATH option.
On some Windows installations, `py` works instead of `python`.

Open a terminal **inside this `volatility_project` folder**. On Windows,
you can open the folder in File Explorer, type `powershell` in its address bar,
and press Enter. Run these commands once:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `.venv` folder keeps this project's packages together. You do not need to
activate it: the commands below call its Python directly. On macOS/Linux,
use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

For the exact main package versions used in the included checks, install from
`requirements-tested.txt` instead of `requirements.txt`. See `VALIDATION.md`
for what was tested and the limits of the synthetic demonstration.

## 3. Try it now with fake data

```powershell
.\.venv\Scripts\python.exe make_demo_data.py
.\.venv\Scripts\python.exe train.py --data data/SYNTHETIC_daily.csv --output my_demo_results
.\.venv\Scripts\python.exe predict.py --data data/SYNTHETIC_daily.csv --models my_demo_results/models.joblib
```

If the synthetic file already exists, skip the first command. Use a **new output
folder name** for each training run; the program protects previous results.

The fake data is invented and is not downloaded from a market. Its scores do
not establish real-world accuracy or satisfy your project's success criterion.
The supplied `demo_results_20_day` folder, if present, is also synthetic only.

## 4. Add your real data

Copy `data/daily_template.csv` to `data/daily.csv` and fill it with one row for
every completed trading session of **one stock**:

```csv
date,close,volume,realized_volatility
2025-01-02,100.50,1250000,0.012
2025-01-03,101.20,1310000,0.015
```

Those two rows only illustrate the format. They are not real observations and
are not enough to train. The columns mean:

| Column | Meaning |
| --- | --- |
| `date` | Session date, formatted `YYYY-MM-DD`. |
| `close` | Closing stock price, greater than zero. |
| `volume` | Total trading volume during that session. |
| `realized_volatility` | That session's realized volatility in decimal units: `0.02` means 2%. |

**Preparing your future 5-minute Alpaca data:** this starter begins after the
bars have been converted into daily observations. The included collector now
does that step for Alpaca; see `COLLECT_DATA.md`. For other sources, sort the bars
within each exchange-local session. One explicit convention is:

```text
first_return = log(first_5_minute_close / session_open)
later_return[j] = log(close[j] / close[j - 1])  # within the same session
daily_realized_variance = sum(all_intraday_returns ** 2)
daily_realized_volatility = sqrt(daily_realized_variance)
daily_volume = sum(bar_volumes)
daily_close = last_bar_close
```

This convention includes the first five-minute interval and **excludes the
overnight return** from realized volatility. Do not mix it with close-to-close
or annualized volatility. The daily-return feature in `data.py` is a separate
close-to-close return and can include the overnight movement.

Use one consistent session definition, price adjustment policy, and data feed.
Check the exchange calendar, holidays, shortened sessions, duplicate bars, and
missing bars before aggregation. Include every expected session in the daily
file: the code cannot detect an entirely missing trading day without a calendar,
and otherwise interprets the next available row as the next trading session.
Do not silently delete a missing session or forward-fill volatility labels.
Do not discard genuine large moves simply because they are unusual.

Unflagged blank or nonfinite values, duplicate dates, negative volume/volatility, and
nonpositive prices are rejected with a message. Rows are sorted automatically.
Collector files may explicitly mark an incomplete session with `session_complete=0`.
That date stays in place with missing values; affected input and target windows
are excluded after their dates have been aligned. Nothing is forward-filled.
Zero volatility is accepted; the neural network and QLIKE use small numerical
floors where a logarithm or division requires a positive number.

Additional CSV columns are ignored deliberately. To add a new model input,
add its calculation in `build_features()` in `data.py`, using only information
available at the forecast time, then train new models.

## 5. Train on real data

```powershell
.\.venv\Scripts\python.exe train.py --data data/daily.csv --output results_real_01
```

To try the small set of alternative settings in `config.py`:

```powershell
.\.venv\Scripts\python.exe train.py --data data/daily.csv --output results_real_02 --tune
```

`--tune` tries two configurations per model family by default. This is a small,
readable starting search, not exhaustive optimization. You can edit or add
setting dictionaries to test other depths, numbers of trees, layers, learning
rates, and batch sizes.

The split is approximately the oldest **60% for training**, the next **20% for
validation**, and the newest **20% for testing**. One example is omitted at each
boundary so a previous range's target date is strictly earlier than the next
range's input date. All models use exactly the same date ranges.

Only training rows fit the models and the network's input scaler. Validation
QLIKE selects settings and the preferred model, including the baseline as a
possible winner. **Ordinary runs evaluate validation only.** The optional
`--validation-only` flag makes this default explicit. The final test is reserved
until you deliberately use `--evaluate-test` after finishing model development:

```powershell
.\.venv\Scripts\python.exe train.py --data data/daily.csv --output results_final --evaluate-test
```

Use the same locked data, settings, and `--tune` choice as your chosen experiment.
This command refits deterministically from those settings; it does not load a
previous experiment automatically. Do not use it for routine experiments.
Saved models remain fitted on the training portion; no automatic refit on
validation or test answers occurs. The network may stop when training loss
stops improving; its built-in random validation split is disabled.

The evaluation simulates a sequence of daily forecasts: after each session
finishes, its observed values can become inputs to the following forecast.
It is **not** a forecast of the entire test period made on its first day.
Avoid changing settings after inspecting test results; that would
turn your test data into another validation set.

By default, date boundaries depend on the number of usable rows. Pass
`--split-from` with an earlier run's `training_info.json` to keep its calendar
boundaries fixed when comparing different inputs, lookbacks, or data coverage.

Outputs in your chosen folder:

- `models.joblib`: fitted models and the input recipe; use with `predict.py`.
- `scores.csv`: baseline and model errors on validation dates; test only with `--evaluate-test`.
- `validation_candidates.csv`: validation scores for each tried configuration.
- `validation_predictions.csv`: validation predictions, actual targets, baseline, and dates.
- `test_predictions.csv`: created **only** with `--evaluate-test`.
- `training_info.json`: settings, input names, date ranges, versions, warnings.

MAE and MSE are calculated on **volatility** in decimal units. QLIKE is
calculated on **variance**, using `ratio = actual_volatility² / predicted_volatility²`
and `mean(ratio - log(ratio) - 1)`. Variances below `1e-12` are floored so zero
values are defined numerically. The 20-day baseline is the mean of the last
20 observed **volatilities**, squared only when calculating QLIKE.

`QLIKE_improvement_percent` is `100 * (baseline_loss - model_loss) / baseline_loss`.
Positive values indicate improvement; a value of at least 5 meets the proposal's
numerical target for that split. It does not establish statistical significance.
The percentage is blank when baseline loss is effectively zero. Percentage
improvement depends on this normalized QLIKE definition; use the same definition
for all comparisons. Tree models fit squared volatility error; the network fits
squared **log-volatility** error. QLIKE is used to compare/select, not as their
training objective.

### Reproduce the first SPY experiment and charts

The collected SPY data is already in `data/alpaca_since2016/daily/SPY.csv`.
Install the optional plotting packages once, then use a **new** report folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-plots.txt
.\.venv\Scripts\python.exe review_training_data.py --data data/alpaca_since2016/daily/SPY.csv --quality data/alpaca_since2016/quality/SPY.csv --symbol SPY --output reports/SPY_pilot_02
.\.venv\Scripts\python.exe train.py --data data/alpaca_since2016/daily/SPY.csv --output reports/SPY_pilot_02/models --validation-only
.\.venv\Scripts\python.exe make_validation_report.py --results reports/SPY_pilot_02/models --review reports/SPY_pilot_02 --symbol SPY
```

Open `SPY_PILOT_REPORT.md` in that folder for the table, chart, dates, settings,
and data-quality limitations. The saved CSV table also opens in Excel. Each
model remains fitted on training data only. The plot/report step does not train
models or read test predictions.

## 6. How the network uses multiple days

Change this single setting in `config.py` before training:

```python
LOOKBACK_DAYS = 20
```

For a forecast made after today's close, the model sees separately named inputs:

```text
today:          return_lag_0,  volatility_lag_0,  log_volume_lag_0
previous day:   return_lag_1,  volatility_lag_1,  log_volume_lag_1
...
19 days ago:    return_lag_19, volatility_lag_19, log_volume_lag_19
                                                        |
                                                        v
                                     next trading day's volatility
```

The network receives all 20 days together as one row, not just their average.
It is a feedforward network with a fixed history window, not an LSTM or a model
with persistent memory. Training on many days is different from giving the
network multiple past days as inputs: this code does both.

Each example has `3 * LOOKBACK_DAYS + 6` inputs. With the default 20-day window,
that means **66 inputs**: 60 historical values plus current log price, 5-day
mean volatility, 20-day mean volatility, relative log volume, and two weekday
values. The default architecture is **66 inputs → 32 neurons → 16 neurons →
1 output**, with the output converted from log volatility back to volatility.

To use 60 trading days instead, set `LOOKBACK_DAYS = 60` and retrain. It will
automatically use 186 inputs. Keep `BASELINE_DAYS = 20` for the proposal's
baseline. Both the baseline and the 5-day summary remain additional inputs
even if you choose a shorter individual-day lookback.

### Add commodity and major ETF prices

Use `--market-data` to give all three models extra market inputs. For SPY,
the original 66 inputs grow to **402**. Each of the 16 other ETFs adds one
log closing price and 20 separate daily log returns. For a forecast after
Tuesday's close, the outside ETF history ends Monday (or the preceding
trading session). This gives the provider's daily bars time to finalize.
SPY's own inputs still include Tuesday's regular session.

| Group | Symbols |
| --- | --- |
| Commodity exposure | GLD (gold), SLV (silver), USO (oil), UNG (natural gas), CPER (copper), DBA (agriculture), URA (uranium and nuclear industry companies) |
| Major equity ETFs | QQQ, DIA, IWM, VTI, EFA, EEM |
| Treasury and sector ETFs | TLT, XLF, XLE |

These inputs are split-adjusted ETF share prices and price returns.
The folder also contains SPY prices for experiments predicting other symbols.
The code skips the target symbol by matching the CSV filename or
`--target-symbol`. Use `--market-symbols GLD SLV QQQ` to try a smaller set.
Dates align to the target's exchange sessions. A missing outside price
excludes forecasts whose input windows need that price.

The ETF files are already in `data/market_context_since2016`. Use a new
output folder to repeat the SPY comparison with the same date boundaries:

```powershell
.\.venv\Scripts\python.exe train.py --data data/alpaca_since2016/daily/SPY.csv --market-data data/market_context_since2016 --split-from reports/SPY_pilot_01/models/training_info.json --output reports/SPY_market_inputs_02/models
```

To load the saved model for a forecast from your latest supplied data:

```powershell
.\.venv\Scripts\python.exe predict.py --data data/alpaca_since2016/daily/SPY.csv --models reports/SPY_market_inputs_01/models/models.joblib --market-data data/market_context_since2016
```

Prediction reads the list of required ETFs and their input windows from the
saved model. For an updated forecast, supply updated target history and ETF
prices. Download ETF prices to a new folder with
`collect_market_prices.py --end YYYY-MM-DD --output data/market_context_updated`.
Each ETF file needs `date` and `close` columns. Use the same data source and
split adjustment as in training. See
[the SPY comparison](VALIDATION.md#spy-with-commodity-and-major-etf-inputs)
for the measured effect of adding these inputs.

### Add daily and Yang–Zhang volatility

The OHLC mode uses the collected **25 ETFs and commodity funds** and **20 major
stocks**, including the target's own OHLC volatility. Each of the 44 other
symbols still contributes a log closing price and 20 daily log returns.
All OHLC inputs end at the previous completed trading session.

| Volatility inputs per symbol | Calculation |
| --- | --- |
| Previous ten sessions, individually | One range-based estimate per day. Square the overnight log return, add the Rogers–Satchell intraday variance, then take the square root. |
| Previous ten sessions, jointly | One Yang–Zhang estimate across the ten sessions, reported in daily units. |
| About one month, six months, and one year | Total range-based and Yang–Zhang volatility at each of 21, 126, and 252 trading sessions. Convert estimated daily variance to period volatility by multiplying by the number of sessions and taking the square root. |

Yang–Zhang estimates daily variance using the sample variance of the overnight
returns, the sample variance of open-to-close returns, and the average
Rogers–Satchell variance. The calculation needs multiple sessions and the close
immediately preceding the window. See [Yang and Zhang (2000)](https://doi.org/10.1086/209650).

The model predicts **the next session's five-minute realized volatility**.
The 20-day comparison baseline still uses that same volatility definition.
The new OHLC volatility inputs replace the original SPY volatility input lags
and averages for this run. Earlier saved models still load their original inputs.

Alpaca daily OHLC files are already in `data/market_ohlc_since2016`.
A 252-session window needs at least **254 dated sessions** for a prediction,
including the earlier close and the one-session delay. At least 90% of a
rolling window must have usable observations. Returns never bridge a missing
session. The models fill any unavailable OHLC-derived inputs using medians
learned on training rows and include indicators for features missing during
training. A feature with no training observations is filled with zero and
flagged as unavailable.

To repeat the validation run with the same holdout dates:

```powershell
.\.venv\Scripts\python.exe train.py --data data/alpaca_since2016/daily/SPY.csv --volatility-data data/market_ohlc_since2016 --split-from reports/SPY_pilot_01/models/training_info.json --output reports/SPY_yang_zhang_02/models
```

To load the saved OHLC model after moving the project or supplying updated bars,
pass the OHLC folder to `--market-data` on the prediction command:

```powershell
.\.venv\Scripts\python.exe predict.py --data data/alpaca_since2016/daily/SPY.csv --models reports/SPY_yang_zhang_01/models/models.joblib --market-data data/market_ohlc_since2016
```

For updated OHLC files, run `collect_market_prices.py --all-symbols --end YYYY-MM-DD --output data/market_ohlc_updated`.
See [the validation results](VALIDATION.md#spy-with-all-market-volatility-inputs).

## 7. Input and training-data limits

| Question | Decision tree / random forest | Neural network |
| --- | --- | --- |
| Maximum input columns? | No small fixed architectural cap; memory and computation are practical limits. | No small fixed cap when creating a new model; this implementation infers input width during fitting. |
| Can input width change after training? | No. Keep the same columns and meaning, or retrain. | No. Its learned weights match the original input width; retrain after changing it. |
| Maximum training rows? | No preset row cap here; memory and time limit what fits. | No preset row cap here; memory, time, layers, and epochs matter. |
| Does batch size limit the dataset? | Not applicable. | No. Batch size 32 means up to 32 examples per weight update, repeated across the training data. |
| Can I append data and update automatically? | This starter retrains from scratch. | This starter also retrains from scratch. It does not implement incremental updates. |

This implementation loads the data into memory and uses CPU training; it does
not stream very large datasets or use a GPU. A bigger forest consumes more
memory, and a wider network has more weights to learn. More input history is
not automatically better, especially when training examples are limited.

With default settings, you need **21 complete daily rows** for one latest
prediction: 20 return inputs need one extra earlier closing price. For training,
`N` complete daily rows give `N - 21` labeled examples. The program requires at
least **100 labeled examples** (121 complete days with these settings) as a
basic guard; that is a coding choice, not a claim that 100 is statistically
sufficient. The 100 examples are then split into training/validation/test.

As a rough planning estimate, 3-5 years at about 252 sessions/year gives
756-1,260 daily rows, or about 735-1,239 labeled examples with this window.
Only about 60% go into fitting. Five-minute source bars give more measurements
of each day's realized volatility, but they do not become independent daily
training targets in this setup. There is no universal minimum sample count
that guarantees a useful model. Keep the network small initially and assess
performance on genuinely later dates.

## 8. Predict after training

Update your real daily CSV through the latest **completed** session, then run:

```powershell
.\.venv\Scripts\python.exe predict.py --data data/daily.csv --models results_real_01/models.joblib
```

Use the same stock, definitions, and units used for training. The script loads
the saved lookback length and feature names; editing `config.py` alone does not
change a saved model. Supply the full history or at least enough recent rows to
fill all saved windows. The final row does not need tomorrow's target value.
The script says "next trading session" rather than guessing a calendar date.

Keep the source files with the model file, and use the same package versions.
Load only joblib files you created or trust. A feature calculation change
requires retraining even if its column name stays the same.

## 9. Checks and scope

Run the included checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_workflow
```

This supplies the requested training-ready model components and evaluation
workflow, plus Alpaca collection and session aggregation. It does not yet implement the optional next-hour
forecast, repeated walk-forward studies, confidence intervals, or a full
ablation study. Those depend on the eventual data and experiment design.

Useful references:

- [scikit-learn decision tree documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [scikit-learn random forest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn neural network documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
- [scikit-learn pipelines and target transformations](https://scikit-learn.org/stable/modules/compose.html)
- [Patton's paper on volatility forecast comparison and loss functions](https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf)
