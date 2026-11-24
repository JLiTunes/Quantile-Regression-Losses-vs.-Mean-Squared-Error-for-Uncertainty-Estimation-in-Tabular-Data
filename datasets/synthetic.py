"""Synthetic heteroscedastic data with exact Gaussian conditional quantiles."""

import numpy as np
from scipy.stats import norm

from config import SEED_START, SYNTH_SAMPLES, SYNTH_SPLITS, TAUS, VAL_FRACTION

from ._utils import scaled_split

BETA = np.array([1.0, -0.8, 0.6, 0.4, -0.3, 0.2])


def _draw(rng, n):
    X = rng.normal(size=(n, BETA.size))
    mu = X @ BETA + 0.4 * np.sin(2.0 * X[:, 0])
    sigma = np.clip(
        0.2 + 0.8 * np.exp(0.5 * X[:, 0] - 0.2 * X[:, 1]),
        0.05,
        10.0,
    )
    y = mu + sigma * rng.normal(size=n)
    return X, y, mu, sigma


def load():
    splits = []
    for fold in range(SYNTH_SPLITS):
        rng = np.random.default_rng(SEED_START + fold + 1)
        X, y, mu, sigma = _draw(rng, SYNTH_SAMPLES)

        perm = rng.permutation(SYNTH_SAMPLES)
        n_test = int(round(SYNTH_SAMPLES * 0.2))
        n_val = int(round(SYNTH_SAMPLES * VAL_FRACTION))
        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        train_idx = perm[n_test + n_val:]

        z = norm.ppf(TAUS)
        q_true_test = (
            mu[test_idx][:, None] + sigma[test_idx][:, None] * z[None, :]
        )
        splits.append(
            scaled_split(X, y, train_idx, val_idx, test_idx, q_true_test)
        )

    return {"name": "synthetic", "splits": splits, "has_true_quantiles": True}
