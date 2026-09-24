# Forecasting models

The app can answer questions about the future ("What will revenue be next
week?") with models trained **offline, once**, from this folder. The server
never trains anything: it only loads the saved models from `models/` and
predicts (`forecasting.py`).

## Pipeline

1. **Data**: the [M4 competition](https://github.com/Mcompetitions/M4-methods)
   daily, weekly, and monthly series (tens of thousands of real series from
   finance, economics, industry, demographics, and more). `m4.py` downloads
   them into `training/data/` (git-ignored, ~200 MB).
2. **Preprocessing**: `forecasting.preprocess`, the same function the app runs
   at inference time, so training and runtime can't drift apart:
   - **Seasonal adjustment**: if a window passes the M4 seasonality test
     (autocorrelation at lag 7 for daily data, 12 for monthly), its cycle is
     removed with classical multiplicative decomposition and put back after
     prediction. M4's daily series have almost no weekly cycle, so without
     this the model flattened the weekly pattern of real daily sales data.
   - **Scaling**: each window is divided by its own mean level, so the model
     learns the shape of how series move, not their size. That's what lets
     one model work on any uploaded dataset.
   - **Features**: the scaled window (28 days / 26 weeks / 24 months), recent
     mean, volatility, trend slope, last change, and the step ahead.
3. **Training** (`train.py`): one LightGBM model per quantile (10%, 50%, 90%),
   giving the forecast and its likely range. 10% of the series are held out
   for early stopping.
4. **Evaluation** (`evaluate.py`): each series is forecast from the end of its
   training data and scored against the values M4 held back, compared with
   naive (repeat the last value) and seasonal-naive (repeat the last cycle).

## Results (M4 test sets, `models/metrics.json`)

| Frequency | Series | Model MASE | Naive | Seasonal naive | vs best baseline | 10-90% coverage |
|---|---|---|---|---|---|---|
| Daily   | 4,227  | 3.276 | 3.278 | 4.128 | +0.1%  | 78.2% |
| Weekly  | 359    | 2.369 | 2.777 | 9.578 | +14.7% | 76.7% |
| Monthly | 48,000 | 0.987 | 1.205 | 1.260 | +18.1% | 76.8% |

Lower MASE is better. Daily only ties naive on M4 because most M4 daily
series are financial random walks, where "tomorrow = today" is close to
unbeatable. Two extra checks cover the business data this app is meant for:

- Synthetic daily sales with a weekly cycle (300 series): model 6.3% error,
  seasonal-naive 5.7%, naive 17.0%. The seasonal adjustment brought the model
  from 15.0% to 6.3%.
- Monthly airline passengers (seaborn `flights`, 1960 held out): model 7.5%
  error, seasonal-naive 10.0%. 83% of the real values fell inside the range.

The likely ranges are a little narrow (77-78% coverage against an 80% target).

## Retraining

From the project root:

```
python training/train.py          # all frequencies, about a minute
python training/evaluate.py
python training/train.py M        # or just one: D, W, M
```

## What the app does with unsuitable data

`forecast()` checks the data before predicting and returns a clear message
instead of a bad forecast when there's no date column, too little history
(28 days / 26 weeks / 24 months), unevenly spaced dates, quarterly or yearly
data, sub-daily timestamps, or a non-numeric column. Horizons are capped at
14 days / 13 weeks / 18 months, with a note to the user.
