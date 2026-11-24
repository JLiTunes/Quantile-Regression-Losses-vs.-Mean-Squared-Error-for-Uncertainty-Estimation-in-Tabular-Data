"""Shared split helper: scaling is always fit on the training rows only."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def scaled_split(X, y, train_idx, val_idx, test_idx, q_true_test=None):
    """Scale features inside the training split and return the data contract tuple."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    q_true = None if q_true_test is None else np.asarray(q_true_test, dtype=np.float64)
    return (
        X_train,
        y[train_idx].copy(),
        X_val,
        y[val_idx].copy(),
        X_test,
        y[test_idx].copy(),
        q_true,
    )
