"""Uncertainty-quality metrics for quantile and Gaussian-interval predictions.

All functions take plain NumPy arrays:

    y_true    : (N,)
    quantiles : (N, T) predicted quantiles, columns aligned with ``taus``
    taus      : (T,)   strictly increasing target quantile levels
"""

import numpy as np


def _prep(y_true, quantiles, taus):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    q = np.asarray(quantiles, dtype=np.float64)
    t = np.asarray(taus, dtype=np.float64).reshape(-1)
    if q.ndim != 2 or q.shape[0] != y.shape[0] or q.shape[1] != t.shape[0]:
        raise ValueError(
            f"quantiles must have shape ({y.shape[0]}, {t.shape[0]}), got {q.shape}"
        )
    return y, q, t


def pinball_loss(y_true, quantiles, taus):
    """Mean pinball (quantile) loss over samples and taus."""
    y, q, t = _prep(y_true, quantiles, taus)
    err = y[:, None] - q
    loss = np.maximum(t[None, :] * err, (t[None, :] - 1.0) * err)
    return float(loss.mean())


def per_tau_pinball(y_true, quantiles, taus):
    """Pinball loss averaged over samples, per tau. Shape (T,)."""
    y, q, t = _prep(y_true, quantiles, taus)
    err = y[:, None] - q
    loss = np.maximum(t[None, :] * err, (t[None, :] - 1.0) * err)
    return loss.mean(axis=0)


def crps_from_quantiles(y_true, quantiles, taus):
    """CRPS approximated from the quantile loss integral.

    CRPS = 2 * integral_0^1 pinball_tau d_tau, computed with the trapezoid
    rule on the sampled tau grid.
    """
    _, _, t = _prep(y_true, quantiles, taus)
    losses = per_tau_pinball(y_true, quantiles, taus)
    if t.size < 2:
        return float(2.0 * losses.mean())
    return float(2.0 * np.sum(0.5 * (losses[1:] + losses[:-1]) * np.diff(t)))


def mae(y_true, point_pred):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(point_pred, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(y - p)))


def rmse(y_true, point_pred):
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(point_pred, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def interval_bounds(quantiles, taus, level=0.8):
    """Central ``level`` interval bounds from the nearest available taus."""
    q = np.asarray(quantiles, dtype=np.float64)
    t = np.asarray(taus, dtype=np.float64).reshape(-1)
    alpha = (1.0 - level) / 2.0
    lo = int(np.argmin(np.abs(t - alpha)))
    hi = int(np.argmin(np.abs(t - (1.0 - alpha))))
    return q[:, lo], q[:, hi]


def picp(y_true, lower, upper):
    """Prediction interval coverage probability (empirical)."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lower, upper):
    """Mean prediction interval width."""
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    return float(np.mean(hi - lo))


def crossing_rate(quantiles):
    """Fraction of adjacent tau pairs that violate monotonicity."""
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 2:
        return 0.0
    return float(np.mean(q[:, 1:] < q[:, :-1]))


def empirical_coverage(y_true, quantiles):
    """Per-tau empirical coverage P(y <= q_tau). Shape (T,)."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    q = np.asarray(quantiles, dtype=np.float64)
    return np.mean(y[:, None] <= q, axis=0)


def true_quantile_mae(quantiles, q_true):
    """MAE against exact conditional quantiles; NaN when unavailable."""
    if q_true is None:
        return float("nan")
    q = np.asarray(quantiles, dtype=np.float64)
    t = np.asarray(q_true, dtype=np.float64)
    if q.shape != t.shape:
        raise ValueError(f"q_true shape {t.shape} != quantiles shape {q.shape}")
    return float(np.mean(np.abs(q - t)))


def evaluate(y_true, quantiles, taus, point_pred, q_true=None, level=0.8):
    """Compute the full scalar metric set for one model run."""
    y, q, t = _prep(y_true, quantiles, taus)
    lower, upper = interval_bounds(q, t, level)
    tag = int(round(level * 100))
    return {
        "pinball": pinball_loss(y, q, t),
        "crps": crps_from_quantiles(y, q, t),
        "mae": mae(y, point_pred),
        "rmse": rmse(y, point_pred),
        f"picp_{tag}": picp(y, lower, upper),
        f"mpiw_{tag}": mpiw(lower, upper),
        "crossing_rate": crossing_rate(q),
        "true_q_mae": true_quantile_mae(q, q_true),
    }
