"""Benchmark figures (calibration, crossing rate, point vs. interval)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import CALIBRATION_LEVEL, MAX_PLOT_POINTS, PLOT_DPI, RESULTS_DIR

sns.set_theme(style="whitegrid", context="notebook")
LEVEL_TAG = int(round(CALIBRATION_LEVEL * 100))


def plot_calibration(coverage_df, out_dir=RESULTS_DIR):
    """Coverage vs. nominal tau for every model, one figure per dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if coverage_df.empty:
        return []

    paths = []
    for dataset, group in coverage_df.groupby("dataset", sort=False):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        sns.lineplot(
            data=group,
            x="tau",
            y="coverage",
            hue="model",
            marker="o",
            errorbar="sd",
            ax=ax,
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", lw=1, label="ideal")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Nominal quantile level (tau)")
        ax.set_ylabel("Empirical coverage  P(y <= q_tau)")
        ax.set_title(f"Calibration - {dataset}")
        ax.legend(fontsize=8, loc="upper left")
        path = out_dir / f"calibration_{dataset}.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_crossing(raw_df, out_dir=RESULTS_DIR):
    """Mean crossing rate per model, grouped by dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw_df.empty:
        return []

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    sns.barplot(
        data=raw_df,
        x="model",
        y="crossing_rate",
        hue="dataset",
        errorbar="sd",
        ax=ax,
    )
    ax.set_title("Quantile crossing rate (mean, sd error bars; lower is better)")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of crossed adjacent quantiles")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="dataset", fontsize=8)
    path = out_dir / "crossing_rate.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]


def plot_interval_predictions(details, out_dir=RESULTS_DIR):
    """Point predictions with central intervals, one panel per model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not details:
        return []

    first_run = {}
    dataset_order = []
    for record in details:
        if record["dataset"] not in dataset_order:
            dataset_order.append(record["dataset"])
        first_run.setdefault((record["dataset"], record["model"]), record)

    paths = []
    for dataset in dataset_order:
        records = [
            record
            for (name, _), record in first_run.items()
            if name == dataset
        ]
        if not records:
            continue

        n = len(records)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False
        )
        for ax in axes.ravel():
            ax.set_visible(False)

        for ax, record in zip(axes.ravel(), records):
            ax.set_visible(True)
            y_true = np.asarray(record["y_true"], dtype=float)
            point = np.asarray(record["point_pred"], dtype=float)
            low = np.asarray(record["q_low"], dtype=float)
            high = np.asarray(record["q_high"], dtype=float)

            m = len(y_true)
            if m > MAX_PLOT_POINTS:
                idx = np.sort(
                    np.random.default_rng(0).choice(m, MAX_PLOT_POINTS, replace=False)
                )
            else:
                idx = np.arange(m)
            p, lo, hi, yt = point[idx], low[idx], high[idx], y_true[idx]

            ax.vlines(p, lo, hi, color="#4472c4", alpha=0.5, lw=1.1)
            ax.scatter(p, yt, s=12, color="#c0504d", zorder=3)

            lower = float(min(lo.min(), yt.min()))
            upper = float(max(hi.max(), yt.max()))
            pad = 0.02 * (upper - lower + 1e-9)
            ax.plot([lower, upper], [lower, upper], "--", color="0.6", lw=1)
            ax.set_xlim(lower - pad, upper + pad)
            ax.set_ylim(lower - pad, upper + pad)
            ax.set_title(record["model"], fontsize=9)
            ax.set_xlabel("Point prediction", fontsize=8)
            ax.set_ylabel("True target", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle(
            f"Point predictions with {LEVEL_TAG}% intervals - {dataset}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = out_dir / f"intervals_{dataset}.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
