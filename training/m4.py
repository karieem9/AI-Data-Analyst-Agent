"""Download and load the M4 competition data (daily, weekly, monthly)."""
import csv
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/{split}/{name}-{split_lower}.csv"
NAMES = {"D": "Daily", "W": "Weekly", "M": "Monthly"}


def _path(freq: str, split: str) -> Path:
    return DATA_DIR / f"{NAMES[freq]}-{split}.csv"


def download(freq: str) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for split in ("train", "test"):
        path = _path(freq, split)
        if path.exists():
            continue
        url = URL.format(split=split.capitalize(), name=NAMES[freq], split_lower=split)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)


def load(freq: str, split: str) -> list[np.ndarray]:
    """One array per series. Read line by line: the files are wide (one
    column per time step, mostly empty), and loading monthly into a
    DataFrame would take ~1 GB for 48,000 short series."""
    download(freq)
    series = []
    with open(_path(freq, split), newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            series.append(np.array([float(v) for v in row[1:] if v not in ("", "NA")]))
    return series
