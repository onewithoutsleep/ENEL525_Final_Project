# Checks performed

Checked on November 10, 2026, using Python 3.12 with:

- NumPy 2.5.4
- pandas 3.0.1
- scikit-learn 1.9.1
- joblib 1.6.0
- SciPy 1.18.1

All seven tests in `test_workflow.py` passed. They check that future data does
not change earlier inputs, targets belong to the following session, date
splits do not overlap future labels, 60-day windows work, QLIKE uses variance,
invalid CSV data is rejected, and the saved network retains its predictions
and training-only scaling.

The command-line workflow also completed for both default training and
`--tune`, including saving all three models and loading them in `predict.py`.
The 20-day demo used 500 synthetic daily rows, producing 479 labeled examples
with 66 inputs each: 287 training, 95 validation, 95 test, and two omitted
boundary examples.

**These initial workflow checks used synthetic data.** They did not establish
real forecasting accuracy. Historical collection and the subsequent real-data
validation experiment are described below.

The files in `demo_results_20_day` are synthetic workflow examples only.
The neural network did not beat the baseline in this synthetic test; the
code does not promise that any model will improve on the baseline with real
data. Hyperparameters should be chosen using real validation data later.

The ordinary `python` command was unavailable in this workspace during
development, so checks used the app's bundled Python and project-local test
dependencies. Follow the installation steps in `README.md` for a normal,
portable Python environment.

## Collection extension

The Alpaca collector adds six offline checks, bringing the combined total to
13 passing tests (`test_collection` and `test_workflow`). These exercise API
pagination, market-session boundaries, first-bar returns, overnight exclusion,
early closes, daylight-saving changes, incomplete/duplicate bars, and preserving
missing-session positions in training windows. These are simulated API/bar
checks; they do not by themselves verify account access or a live download.
The added connection-recovery check verifies retries after a dropped HTTP
connection and reading compressed API responses.
Daily aggregation was then changed to operate on whole columns for speed. Its
daily outputs and quality flags matched the original calculation on the real
2016 SPY sample within floating-point tolerance, and all 13 checks still passed.
The optimized calculation also matched the previously saved SPY daily values
and quality flags over all 2,729 requested sessions from 2016 through 2026.

The user's Alpaca connection subsequently returned real historical SIP bars
from January 2016. Collection was expanded to January 1, 2016 through November
9, 2026 for 40 symbols. The final collection status and actual coverage are
recorded in `data/alpaca_since2016/collection_info.json` and
`data/alpaca_since2016/collection_summary.csv`. A completed verification creates
`data/alpaca_since2016/COLLECTION_REPORT.md`; a live download alone does not
establish model accuracy.

## Completed real-data collection

All 40 requested symbols finished successfully. The saved source data contains
440 symbol/year files and 13,205,588 five-minute bars. Final verification checked
every daily file against the 2,729-session calendar, checked finite usable
inputs and next-session target dates, and checked the daily-data ZIP's integrity.

There are 108,841 complete regular sessions across the 40 datasets and 104,188
usable examples after the default 20-day lookback and quality exclusions.
Each symbol has 2,437-2,645 usable examples. All 40 exceed the starter's minimum.
The report documents the exclusion of incomplete sessions and the important
limitation for the March 2020 market-wide halt dates. Real-data models have
not been fitted or evaluated by the collection/verification steps.

## First SPY validation experiment — November 11, 2026

`reports/SPY_pilot_01/SPY_PILOT_REPORT.md` contains the first real-data comparison.
The original configurations were fitted on 1,581 SPY training examples and
compared on 526 validation forecasts, targeting August 29, 2022 through
October 1, 2024. All models used 20 trading days of history and 66 inputs.
No parameter search was performed. No training warnings were recorded; the
neural network stopped after 96 passes through the training data.

Validation QLIKE was 0.233165 for the 20-day average, 0.192666 for the decision
tree, 0.170513 for the random forest, and 1.161029 for the neural network.
The forest's QLIKE was 26.87% below the baseline and the tree's was 17.37% below.
The network's QLIKE was 397.94% above the baseline. These are development
results, not final-test performance or evidence of statistical significance.

The 527 final-test examples, targeting October 3, 2024 through November 9, 2026,
were not predicted or scored. `train.py` now defaults to validation-only;
`--evaluate-test` is a deliberate opt-in for the final assessment. The default
run creates `validation_predictions.csv` and no `test_predictions.csv`.

All **15 offline tests passed**: nine workflow checks and six collection checks.
The two added workflow checks guard every model prediction and score calculation
against unauthorized date ranges, verify training-only fits, and confirm that
test results appear only with explicit opt-in. Both checks use synthetic data.
The plot/report workflow used Matplotlib 3.11.2. Both generated charts were
visually inspected for labels, units, dates, and legibility.

The training-data review covered 1,674 calendar sessions, including six flagged
incomplete dates. Their affected history/target windows removed 72 additional
examples after the ordinary history requirement. The strict exclusion policy
was retained. Four flagged dates were March 2020 market-wide halt sessions;
this experiment does not establish performance on those excluded crisis days.

The report folder preserves settings, date boundaries, training diagnostics,
input/source fingerprints, predictions, and the fitted models. The final
artifact verification is recorded in `reports/SPY_pilot_01/verification.json`.

## SPY with commodity and major ETF inputs

The November 20, 2026 experiment compares the original 66 inputs with 402
inputs using the same 1,581 training examples, 526 validation forecasts,
and model settings. The new inputs use each outside ETF's previous-session
log closing price and 20 daily log returns. Alpaca supplied 2,729 daily prices
per ETF across all 17 requested ETFs from January 4, 2016 to November 9, 2026.
SPY serves as the target and is excluded from the 16 outside ETFs in this run.

| Model | Original validation QLIKE | With ETF inputs | Original MAE (percentage points) | With ETF inputs |
| --- | ---: | ---: | ---: | ---: |
| 20-day average | 0.2332 | 0.2332 | 0.1784 | 0.1784 |
| Decision tree | 0.1927 | 0.1973 | 0.1658 | 0.1681 |
| Random forest | 0.1705 | 0.1697 | 0.1507 | 0.1529 |
| Neural network | 1.1610 | 6.2917 | 0.2494 | 0.6454 |

Lower errors are better. The expanded random forest's QLIKE was 0.5% lower
than its original score and 27.2% below the 20-day average. Its MAE increased
slightly. The expanded tree and neural network scored worse. In this trial,
the added ETF inputs did not help the neural network with its original
settings. The 527 final-test forecasts are reserved for the final assessment.

The expanded run's fitted models, input list, exact validation scores, and
forecasts are in `reports/SPY_market_inputs_01/models`. ETF price files and
coverage are in `data/market_context_since2016`. All 22 automated checks
passed, including checks for date alignment, price availability, preserved
holdout boundaries, and reuse of the saved input recipe in prediction.

## SPY with all market volatility inputs

The expanded run uses **1,733 data inputs** from target history, 25 ETFs and
commodity funds, and 20 major stocks. It uses ten individual daily volatility
estimates, a 10-session Yang–Zhang estimate, and both volatility estimates at
21, 126, and 252 sessions for each symbol. Forecasts use daily OHLC bars
through the preceding session. The target remains next-session five-minute
realized volatility.

Alpaca SIP supplied split-adjusted daily open, high, low, and close prices
from January 4, 2016 to November 9, 2026. Forty-four symbols have all 2,729
sessions. CPER has 2,682 usable sessions after excluding 47 unusable bars.
Long-window estimates require at least 90% usable sessions and cannot bridge a
missing day. The models fill missing inputs with training medians and include
missingness indicators learned during training.

| Model | ETF price-only QLIKE | All-market OHLC QLIKE | All-market OHLC MAE (percentage points) |
| --- | ---: | ---: | ---: |
| 20-day average | 0.2332 | 0.2332 | 0.1784 |
| Decision tree | 0.1973 | 0.3553 | 0.2240 |
| Random forest | 0.1697 | 0.1938 | 0.1691 |
| Neural network | 6.2917 | 6.0696 | 1.2918 |

The random forest has the lowest QLIKE, 16.9% below the 20-day average.
The neural network's QLIKE improved slightly from the ETF price-only run,
while its MAE increased from 0.6454 to 1.2918 percentage points.
Both experiments use the same 526 validation dates and original model settings.
The year window reduces the training sample from 1,581 to 1,369 examples.
The comparison reflects both the changed inputs and the shorter usable training
history. The 527 final-test examples remain reserved.

The trained models, input list, raw validation metrics, and forecasts are in
`reports/SPY_yang_zhang_01/models`. The 27 automated checks cover the
Yang–Zhang formula, earlier-close alignment, missing sessions, window timing,
the forecast target, and existing model workflows.
