"""
Forecasting with models trained offline on the M4 competition data (see
training/). At runtime this module only loads the saved LightGBM models and
predicts -- nothing here ever trains on the user's data.

The models are "global": each series is scaled by its own recent level
before it reaches the model, so the model learns the *shape* of how series
move (trend, seasonality, mean reversion), not the magnitude of any one
dataset. That's what lets a model trained once work on any uploaded file.

preprocess() is the single source of truth for the preprocessing: training
imports it from here, so training and inference can never drift apart.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"
QUANTILES = (0.1, 0.5, 0.9)  # lower bound, forecast, upper bound

# window: how many past points the model sees. horizon: the furthest step
# ahead it was trained to predict (the M4 competition's horizons). season:
# the cycle removed before the model sees the data (None: a yearly cycle of
# 52 weeks doesn't fit in the window, so weekly data isn't adjusted).
FREQS = {
    "D": {"name": "daily", "unit": "days", "window": 28, "horizon": 14, "season": 7},
    "W": {"name": "weekly", "unit": "weeks", "window": 26, "horizon": 13, "season": None},
    "M": {"name": "monthly", "unit": "months", "window": 24, "horizon": 18, "season": 12},
}

# Daily business data with no weekends is ~28.6% missing days -- still a
# real daily series. Past this, the gaps are the data, not noise.
MAX_MISSING_FRACTION = 0.3
HISTORY_SHOWN = 4  # windows of history to include alongside the forecast
FORECAST_COLUMNS = ["actual", "forecast", "lower", "upper"]


class ForecastError(Exception):
    """The data can't be forecast (no dates, too short, irregular...). The
    message is written for the user; it's not a code bug worth a retry."""


# ---------------------------------------------------------------------------
# Preprocessing shared with training/train.py and training/evaluate.py
# ---------------------------------------------------------------------------

def preprocess(windows, freq: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """The full preprocessing pipeline. Returns (model rows, multiplier):
    the model predicts scale-free, season-free values, and
    prediction * multiplier (n, horizon) turns them back into real units.
    Training divides its targets by the same multiplier."""
    windows = np.asarray(windows, dtype=float)
    past_factors, future_factors = seasonal_factors(windows, FREQS[freq]["season"], horizon)
    features, scale = make_features(windows / past_factors)
    return with_horizons(features, horizon), scale[:, None] * future_factors


def seasonal_factors(windows: np.ndarray, season: int | None, horizon: int):
    """Classical multiplicative seasonal factors per window, for the window
    itself (n, window) and the `horizon` steps after it (n, horizon). A
    window gets factors only if it passes the M4 competition's seasonality
    test (autocorrelation at the seasonal lag, 90% level) and is all
    positive; every other window gets factors of 1 (left unchanged).

    Removing the cycle before the model sees the data means the model never
    has to learn it -- which matters because M4's daily series are mostly
    financial and have almost no weekly cycle to learn from, while real
    daily business data (sales, visits) usually has a strong one."""
    n, length = windows.shape
    past = np.ones((n, length))
    future = np.ones((n, horizon))
    if not season or length < 2 * season:
        return past, future

    centered = windows - windows.mean(axis=1, keepdims=True)
    denom = (centered ** 2).sum(axis=1)
    denom = np.where(denom > 0, denom, 1.0)
    acf = np.stack([(centered[:, k:] * centered[:, :-k]).sum(axis=1) / denom for k in range(1, season + 1)], axis=1)
    limit = 1.645 * np.sqrt((1 + 2 * (acf[:, :-1] ** 2).sum(axis=1)) / length)
    seasonal = (windows > 0).all(axis=1) & (acf[:, -1] > limit)
    if not seasonal.any():
        return past, future

    x = windows[seasonal]
    kernel = np.ones(season) / season if season % 2 else np.r_[0.5, np.ones(season - 1), 0.5] / season
    trend = np.stack([np.convolve(row, kernel, mode="valid") for row in x])
    half = len(kernel) // 2
    ratios = x[:, half:half + trend.shape[1]] / trend
    phase = np.arange(half, half + trend.shape[1]) % season
    factors = np.stack([ratios[:, phase == p].mean(axis=1) for p in range(season)], axis=1)
    factors /= factors.mean(axis=1, keepdims=True)

    past[seasonal] = factors[:, np.arange(length) % season]
    future[seasonal] = factors[:, (length - 1 + np.arange(1, horizon + 1)) % season]
    return past, future


def make_features(windows) -> tuple[np.ndarray, np.ndarray]:
    """windows: (n, window) raw past values, oldest first.
    Returns (features (n, F), scale (n,)). Every feature is scale-free."""
    windows = np.asarray(windows, dtype=float)
    scale = np.abs(windows).mean(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    x = windows / scale[:, None]

    length = x.shape[1]
    t = np.arange(length) - (length - 1) / 2
    slope = (x * t).sum(axis=1) / (t ** 2).sum()
    recent = x[:, -max(2, length // 4):]
    features = np.column_stack([
        x,
        recent.mean(axis=1),
        x.std(axis=1),
        slope,
        x[:, -1] - x[:, -2],
    ])
    return features, scale


def with_horizons(features: np.ndarray, horizon: int) -> np.ndarray:
    """One row per (sample, step ahead): the sample's features plus the step
    number, ordered sample-major -- matching targets.ravel() of (n, horizon)."""
    steps = np.tile(np.arange(1, horizon + 1), len(features))
    return np.column_stack([np.repeat(features, horizon, axis=0), steps])


def model_path(freq: str, quantile: float) -> Path:
    return MODELS_DIR / f"forecast_{freq}_q{round(quantile * 100)}.txt"


@lru_cache(maxsize=None)
def _load_models(freq: str) -> dict:
    import lightgbm as lgb

    boosters = {}
    for q in QUANTILES:
        path = model_path(freq, q)
        if not path.exists():
            raise ForecastError("The forecasting models are missing. Run `python training/train.py` first.")
        boosters[q] = lgb.Booster(model_file=str(path))
    return boosters


def predict_windows(windows, freq: str, horizon: int) -> dict:
    """Predict `horizon` steps after each window. Returns {quantile: (n, horizon)}
    in the original units, with quantiles sorted so lower <= forecast <= upper."""
    rows, multiplier = preprocess(windows, freq, horizon)
    boosters = _load_models(freq)
    stacked = np.stack([
        boosters[q].predict(rows).reshape(len(multiplier), horizon) for q in QUANTILES
    ])
    stacked = np.sort(stacked, axis=0) * multiplier[None]
    return {q: stacked[i] for i, q in enumerate(QUANTILES)}


# ---------------------------------------------------------------------------
# Runtime entry point, exposed to generated code in the sandbox
# ---------------------------------------------------------------------------

def forecast(series, periods: int | None = None) -> pd.DataFrame:
    """
    Forecast a numeric series indexed by date, `periods` steps ahead.

    Returns a DataFrame indexed by date with columns actual/forecast/lower/
    upper: recent history rows carry `actual`, future rows carry the
    forecast and its 10%-90% range. Raises ForecastError, with a message
    for the user, when the data can't be forecast.
    """
    series = _as_series(series)
    dates, values = _parse_dates(series)
    freq, period_alias = _infer_frequency(dates)
    cfg = FREQS[freq]

    s = pd.Series(values, index=dates.to_period(period_alias))
    if s.index.duplicated().any():
        raise ValueError(
            f"Several values fall in the same {cfg['name'][:-2]} period. Aggregate first, "
            "e.g. df.groupby('date')['col'].sum(), then pass that to forecast()."
        )
    s = s.sort_index()
    full = pd.period_range(s.index[0], s.index[-1], freq=period_alias)
    missing = len(full) - len(s)
    if missing / len(full) > MAX_MISSING_FRACTION:
        raise ForecastError(
            f"Can't forecast: {missing} of {len(full)} {cfg['unit']} in this range have no data, "
            "so the series is too irregular to forecast reliably."
        )
    s = s.reindex(full).interpolate(limit_direction="both")

    if len(s) < cfg["window"]:
        raise ForecastError(
            f"Can't forecast: {cfg['name']} data needs at least {cfg['window']} {cfg['unit']} "
            f"of history, and this has {len(s)}."
        )

    notes = []
    if missing:
        notes.append(f"{missing} missing {cfg['unit']} were filled in by interpolation.")
    if periods is None:
        periods = cfg["horizon"]
    periods = int(periods)
    if periods < 1:
        raise ValueError("periods must be at least 1.")
    if periods > cfg["horizon"]:
        notes.append(
            f"Asked for {periods} {cfg['unit']}, but {cfg['name']} forecasts are limited to "
            f"{cfg['horizon']} {cfg['unit']} ahead, so this shows {cfg['horizon']}."
        )
        periods = cfg["horizon"]

    if s.nunique() == 1:
        level = float(s.iloc[-1])
        lower = mid = upper = np.full(periods, level)
    else:
        preds = predict_windows(s.values[-cfg["window"]:][None, :], freq, periods)
        lower, mid, upper = (preds[q][0] for q in QUANTILES)
        if (s.values >= 0).all():  # counts, revenue, etc. can't go negative
            lower, mid, upper = (np.clip(a, 0, None) for a in (lower, mid, upper))

    history = s.iloc[-HISTORY_SHOWN * cfg["window"]:]
    future = pd.period_range(s.index[-1] + 1, periods=periods, freq=period_alias)
    nan_hist = np.full(len(history), np.nan)
    nan_future = np.full(periods, np.nan)
    out = pd.concat([
        pd.DataFrame({"actual": history.values, "forecast": nan_hist, "lower": nan_hist, "upper": nan_hist},
                     index=_to_timestamps(history.index, freq)),
        pd.DataFrame({"actual": nan_future, "forecast": mid, "lower": lower, "upper": upper},
                     index=_to_timestamps(future, freq)),
    ])
    out.index.name = series.index.name or "date"
    out.attrs["forecast"] = {"frequency": cfg["name"], "periods": periods, "notes": notes}
    return out


def is_forecast_frame(result) -> bool:
    return isinstance(result, pd.DataFrame) and list(result.columns) == FORECAST_COLUMNS


def _as_series(data) -> pd.Series:
    if isinstance(data, pd.DataFrame):
        from charts import looks_like_dates

        numeric = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        date_cols = [c for c in data.columns if c not in numeric and looks_like_dates(data[c])]
        if len(numeric) == 1 and len(date_cols) == 1:
            data = data.set_index(date_cols[0])[numeric[0]]
        elif len(numeric) == 1:
            data = data[numeric[0]]
        else:
            raise ValueError(
                "forecast() needs one numeric column indexed by date, "
                "e.g. forecast(df.groupby('date')['revenue'].sum(), 7)."
            )
    if not isinstance(data, pd.Series):
        raise ValueError("forecast() needs a pandas Series indexed by date.")
    if not pd.api.types.is_numeric_dtype(data):
        raise ForecastError(f"Can't forecast '{data.name}': forecasting only works on numeric values.")
    return data


def _parse_dates(series: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    no_dates = ForecastError(
        "Can't forecast: this data has no date or time to order it by. "
        "Forecasting needs a date column."
    )
    idx = series.index
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()
    if pd.api.types.is_datetime64_any_dtype(idx):
        dates = pd.DatetimeIndex(idx)
    elif pd.api.types.is_numeric_dtype(idx):
        if len(idx) and pd.api.types.is_integer_dtype(idx) and idx.min() >= 1800 and idx.max() <= 2200:
            raise ForecastError(
                "Can't forecast yearly data. Forecasting works on daily, weekly, or monthly data."
            )
        raise no_dates
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(idx.astype(str), errors="coerce", format="mixed"))
        if dates.isna().mean() > 0.2:
            raise no_dates
    if dates.tz is not None:
        dates = dates.tz_localize(None)

    keep = dates.notna() & series.notna().values
    dates, values = dates[keep], series.values[keep].astype(float)
    if dates.duplicated().any():
        raise ValueError(
            "The series has several values per date. Aggregate first, "
            "e.g. df.groupby('date')['col'].sum(), then pass that to forecast()."
        )
    if len(dates) < 3:
        raise ForecastError(f"Can't forecast: there are only {len(dates)} dated values.")
    order = np.argsort(dates.values)
    return dates[order], values[order]


# (name, smallest gap, largest gap in days) for each spacing we recognize.
SPACINGS = [("D", 1, 1), ("W", 6, 8), ("M", 28, 31), ("Q", 89, 92), ("Y", 365, 366)]
MIN_REGULAR_SHARE = 0.6  # of gaps that must match the spacing; the rest are missing dates


def _infer_frequency(dates: pd.DatetimeIndex) -> tuple[str, str]:
    gaps = np.diff(dates.values).astype("timedelta64[s]").astype(float) / 86400
    if np.median(gaps) < 1:
        raise ForecastError(
            "Can't forecast data recorded more often than daily. Ask for a daily total first."
        )
    # The median alone can't tell "monthly" from random dates that happen
    # to average a month apart, so most gaps must actually be that size.
    shares = {name: np.mean((gaps >= lo - 0.01) & (gaps <= hi + 0.01)) for name, lo, hi in SPACINGS}
    name = max(shares, key=shares.get)
    if shares[name] < MIN_REGULAR_SHARE:
        raise ForecastError(
            "Can't forecast: the dates aren't evenly spaced (not daily, weekly, or monthly), "
            "so there's no regular series to extend."
        )
    if name == "Q":
        raise ForecastError("Can't forecast quarterly data. Forecasting works on daily, weekly, or monthly data.")
    if name == "Y":
        raise ForecastError("Can't forecast yearly data. Forecasting works on daily, weekly, or monthly data.")
    if name == "W":
        # Anchor weeks on the weekday the data actually uses, so dates
        # round-trip unchanged instead of shifting to Sundays.
        return "W", f"W-{dates[-1].day_name()[:3].upper()}"
    return name, name


def _to_timestamps(periods: pd.PeriodIndex, freq: str) -> pd.DatetimeIndex:
    # Weeks are labeled by their anchor day (the data's own weekday); days
    # and months by their start.
    return periods.to_timestamp(how="end").normalize() if freq == "W" else periods.to_timestamp()
