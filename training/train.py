"""
Train the forecasting models offline. Run once from the project root:

    python training/train.py

For each frequency (daily, weekly, monthly) this samples windows from the
M4 training series, preprocesses them with forecasting.preprocess (the
same function the app uses at runtime), and trains one LightGBM quantile
model per quantile. The models are saved as text files in models/; the app
only ever loads them.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import m4  # noqa: E402
from forecasting import FREQS, MODELS_DIR, QUANTILES, model_path, preprocess  # noqa: E402

SEED = 0
# Windows sampled per series: fewer series -> more windows each, so every
# frequency gets a similar amount of training data.
WINDOWS_PER_SERIES = {"D": 10, "W": 60, "M": 2}
MAX_SERIES = {"D": None, "W": None, "M": 20000}
VALID_FRACTION = 0.1  # of series, held out for early stopping
TARGET_CLIP = 10.0  # scaled targets beyond this are data glitches, not signal
PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": SEED,
}
MAX_ROUNDS = 600


def sample_windows(series: list[np.ndarray], window: int, horizon: int, per_series: int, rng):
    """Random (past window, next `horizon` values) pairs from each series."""
    pasts, futures = [], []
    for y in series:
        last_end = len(y) - horizon
        if last_end < window:
            continue
        ends = rng.integers(window, last_end + 1, size=min(per_series, last_end - window + 1))
        for end in ends:
            pasts.append(y[end - window:end])
            futures.append(y[end:end + horizon])
    return np.array(pasts), np.array(futures)


def build_rows(pasts, futures, freq, horizon):
    X, multiplier = preprocess(pasts, freq, horizon)
    y = np.clip(futures / multiplier, -TARGET_CLIP, TARGET_CLIP).ravel()
    return X, y


def train_frequency(freq: str, rng) -> dict:
    cfg = FREQS[freq]
    window, horizon = cfg["window"], cfg["horizon"]
    series = m4.load(freq, "train")
    if MAX_SERIES[freq] and len(series) > MAX_SERIES[freq]:
        idx = rng.choice(len(series), MAX_SERIES[freq], replace=False)
        series = [series[i] for i in idx]

    order = rng.permutation(len(series))
    n_valid = int(len(series) * VALID_FRACTION)
    valid = [series[i] for i in order[:n_valid]]
    train = [series[i] for i in order[n_valid:]]

    X_tr, y_tr = build_rows(*sample_windows(train, window, horizon, WINDOWS_PER_SERIES[freq], rng), freq, horizon)
    X_va, y_va = build_rows(*sample_windows(valid, window, horizon, WINDOWS_PER_SERIES[freq], rng), freq, horizon)
    print(f"[{cfg['name']}] {len(train)} train / {len(valid)} valid series -> "
          f"{len(X_tr):,} train rows, {len(X_va):,} valid rows, {X_tr.shape[1]} features")

    info = {"series": len(series), "train_rows": len(X_tr), "valid_rows": len(X_va), "rounds": {}}
    for q in QUANTILES:
        start = time.time()
        booster = lgb.train(
            {**PARAMS, "objective": "quantile", "alpha": q},
            lgb.Dataset(X_tr, y_tr),
            num_boost_round=MAX_ROUNDS,
            valid_sets=[lgb.Dataset(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        booster.save_model(str(model_path(freq, q)), num_iteration=booster.best_iteration)
        info["rounds"][str(q)] = booster.best_iteration
        print(f"  q={q}: {booster.best_iteration} rounds, {time.time() - start:.0f}s")
    return info


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    freqs = sys.argv[1:] or list(FREQS)
    info = {freq: train_frequency(freq, rng) for freq in freqs}
    meta_path = MODELS_DIR / "training_info.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    existing.update(info)
    meta_path.write_text(json.dumps(existing, indent=2))
    print(f"Saved models to {MODELS_DIR}. Next: python training/evaluate.py")


if __name__ == "__main__":
    main()
