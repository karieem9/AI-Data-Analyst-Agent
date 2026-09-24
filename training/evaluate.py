"""
Evaluate the trained models on the official M4 test sets, against simple
baselines. Run after training, from the project root:

    python training/evaluate.py

Each series is forecast from the end of its training data and compared
with the real values the M4 organizers held back. The model is only worth
shipping if it beats the naive baselines here; results go to
models/metrics.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import m4  # noqa: E402
from forecasting import FREQS, MODELS_DIR, QUANTILES, predict_windows  # noqa: E402

# Seasonal period used to scale MASE, per the M4 competition's own rules,
# and the season length for the seasonal-naive baseline.
MASE_PERIOD = {"D": 1, "W": 1, "M": 12}
SEASON = {"D": 7, "W": 52, "M": 12}


def smape(actual, pred):
    denom = np.abs(actual) + np.abs(pred)
    return np.nanmean(np.where(denom > 0, 200 * np.abs(actual - pred) / denom, 0.0))


def mase(train, actual, pred, m):
    scales = np.array([np.mean(np.abs(y[m:] - y[:-m])) for y in train])
    errors = np.abs(actual - pred).mean(axis=1)
    ok = scales > 0
    return float(np.mean(errors[ok] / scales[ok]))


def seasonal_naive(train, horizon, season):
    out = []
    for y in train:
        if len(y) < season:
            out.append(np.repeat(y[-1], horizon))
        else:
            out.append(np.resize(y[-season:], horizon))
    return np.array(out)


def evaluate_frequency(freq: str) -> dict:
    cfg = FREQS[freq]
    window, horizon = cfg["window"], cfg["horizon"]
    train = m4.load(freq, "train")
    actual = np.array(m4.load(freq, "test"))

    preds = predict_windows(np.array([y[-window:] for y in train]), freq, horizon)
    model = preds[0.5]
    naive = np.array([np.repeat(y[-1], horizon) for y in train])
    snaive = seasonal_naive(train, horizon, SEASON[freq])

    m = MASE_PERIOD[freq]
    lower, upper = preds[QUANTILES[0]], preds[QUANTILES[-1]]
    result = {
        "series": len(train),
        "horizon": horizon,
        "model": {"sMAPE": smape(actual, model), "MASE": mase(train, actual, model, m)},
        "naive": {"sMAPE": smape(actual, naive), "MASE": mase(train, actual, naive, m)},
        "seasonal_naive": {"sMAPE": smape(actual, snaive), "MASE": mase(train, actual, snaive, m)},
        "interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
    }
    best_baseline = min(result["naive"]["MASE"], result["seasonal_naive"]["MASE"])
    result["beats_baselines"] = result["model"]["MASE"] < best_baseline
    result["improvement_vs_best_baseline_%"] = round(100 * (1 - result["model"]["MASE"] / best_baseline), 1)
    return result


def main():
    freqs = sys.argv[1:] or list(FREQS)
    metrics_path = MODELS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for freq in freqs:
        r = evaluate_frequency(freq)
        metrics[freq] = r
        name = FREQS[freq]["name"]
        print(f"\n[{name}] {r['series']} series, horizon {r['horizon']}")
        print(f"  {'':16}{'sMAPE':>8}{'MASE':>8}")
        for label in ("model", "naive", "seasonal_naive"):
            print(f"  {label:16}{r[label]['sMAPE']:8.2f}{r[label]['MASE']:8.3f}")
        target = QUANTILES[-1] - QUANTILES[0]
        print(f"  10-90% interval coverage: {r['interval_coverage']:.1%} (target {target:.0%})")
        verdict = "beats" if r["beats_baselines"] else "DOES NOT beat"
        print(f"  Model {verdict} the best baseline on MASE ({r['improvement_vs_best_baseline_%']:+}%)")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {metrics_path}")


if __name__ == "__main__":
    main()
