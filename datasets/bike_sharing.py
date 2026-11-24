"""UCI Bike Sharing (hourly): chronological TimeSeriesSplit, no temporal leakage."""

import io
import urllib.request
import zipfile

import numpy as np

from config import BIKE_SPLITS, DATA_DIR, VAL_FRACTION

from ._utils import scaled_split

URLS = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/"
    "Bike-Sharing-Dataset.zip",
)
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset"
DROP_COLUMNS = ("instant", "dteday", "casual", "registered")


def _download_hour_csv(path):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in URLS:
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 (benchmark loader)"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                hour_name = next(
                    n for n in archive.namelist() if n.lower().endswith("hour.csv")
                )
                path.write_bytes(archive.read(hour_name))
            return
        except Exception as exc:  # network, zip or timeout problems
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Could not download the UCI Bike Sharing dataset automatically.\n  "
        + "\n  ".join(errors)
        + f"\nDownload hour.csv manually from {DATASET_PAGE} and place it at "
        f"{path}."
    )


def load():
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit

    path = DATA_DIR / "hour.csv"
    if not path.exists():
        _download_hour_csv(path)

    df = pd.read_csv(path)
    missing = [c for c in DROP_COLUMNS if c not in df.columns]
    if missing or "cnt" not in df.columns:
        raise RuntimeError(f"{path} does not look like UCI hour.csv")
    df = df.drop(columns=list(DROP_COLUMNS))

    y = df.pop("cnt").to_numpy(dtype=np.float64)
    X = df.to_numpy(dtype=np.float64)

    splits = []
    tscv = TimeSeriesSplit(n_splits=BIKE_SPLITS)
    for train_pool, test_idx in tscv.split(X):
        n_val = max(1, int(round(len(train_pool) * VAL_FRACTION)))
        train_idx = train_pool[:-n_val]
        val_idx = train_pool[-n_val:]
        splits.append(scaled_split(X, y, train_idx, val_idx, test_idx))

    return {"name": "bike_sharing", "splits": splits, "has_true_quantiles": False}
