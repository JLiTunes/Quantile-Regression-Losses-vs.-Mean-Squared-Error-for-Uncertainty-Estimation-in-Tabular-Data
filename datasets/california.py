"""California housing: K-Fold CV with a validation slice inside each fold."""

import numpy as np

from config import N_FOLDS, SEED_START, VAL_FRACTION

from ._utils import scaled_split


def load():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import KFold

    bunch = fetch_california_housing()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target, dtype=np.float64)

    rng = np.random.default_rng(SEED_START + 42)
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    splits = []
    for train_pool, test_idx in kfold.split(X):
        pool = rng.permutation(train_pool)
        n_val = max(1, int(round(len(pool) * VAL_FRACTION)))
        val_idx = pool[:n_val]
        train_idx = pool[n_val:]
        splits.append(scaled_split(X, y, train_idx, val_idx, test_idx))

    return {"name": "california", "splits": splits, "has_true_quantiles": False}
