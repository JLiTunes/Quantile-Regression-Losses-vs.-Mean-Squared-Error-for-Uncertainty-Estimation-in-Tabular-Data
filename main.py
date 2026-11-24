"""Quantile regression vs. MSE uncertainty benchmark.

Run ``python main.py`` for the interactive menu, or bypass it with flags::

    python main.py --datasets synthetic --models nn_qr_raw,nn_qr_mono --seeds 1
"""

import argparse
import sys

import numpy as np
import pandas as pd

import config
from datasets import DATASETS, resolve_choice as resolve_datasets
from metrics import empirical_coverage, evaluate, interval_bounds

_MODELS_PACKAGE = config.MODELS_ROOT / "models"
if not _MODELS_PACKAGE.is_dir():
    raise SystemExit(
        f"Model package not found: {_MODELS_PACKAGE}\n"
        "Set BENCH_MODELS_ROOT to the folder that contains the 'models' package."
    )
sys.path.insert(0, str(config.MODELS_ROOT))
from models import MODELS, resolve_choice as resolve_models

LEVEL_TAG = int(round(config.CALIBRATION_LEVEL * 100))
AGGREGATED_METRICS = [
    "pinball",
    "crps",
    "mae",
    "rmse",
    f"picp_{LEVEL_TAG}",
    f"mpiw_{LEVEL_TAG}",
    "crossing_rate",
    "true_q_mae",
    "train_time",
    "infer_time",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark quantile regression losses against MSE-based "
            "uncertainty intervals on tabular data."
        )
    )
    parser.add_argument(
        "--datasets",
        help="comma separated dataset IDs/names (e.g. 1,3), or 'all'",
    )
    parser.add_argument(
        "--models",
        help="comma separated model IDs/names (e.g. 2,5), or 'all'",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        help=f"number of random seeds (default {config.DEFAULT_SEEDS}; 1 = smoke test)",
    )
    return parser.parse_args(argv)


def print_menu(title, registry):
    print(f"\n{title}")
    for key, spec in registry.items():
        print(f"  {key}) {spec['name']:<12s} {spec['description']}")


def ask(prompt, default):
    try:
        text = input(prompt).strip()
    except EOFError:
        return default
    return text or default


def run_benchmark(dataset_ids, model_ids, n_seeds):
    """Run every model x split x seed combination, collecting metrics."""
    records = []
    coverage_rows = []
    details = []
    failures = []
    reported_errors = set()
    seeds = list(range(config.SEED_START, config.SEED_START + n_seeds))

    for dataset_id in dataset_ids:
        dataset_spec = DATASETS[dataset_id]
        dataset_name = dataset_spec["name"]
        print(f"\n== {dataset_name} ==")

        try:
            bundle = dataset_spec["loader"]()
        except Exception as exc:
            print(f"!! skipping dataset {dataset_name}: {type(exc).__name__}: {exc}")
            failures.append(
                (dataset_name, "-", f"{type(exc).__name__}: {exc}")
            )
            continue

        splits = bundle["splits"]
        print(
            f"   {len(splits)} split(s) | "
            f"true quantiles: {bundle['has_true_quantiles']}"
        )

        for split_idx, split in enumerate(splits, start=1):
            (
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                q_true_test,
            ) = split

            for model_id in model_ids:
                model_spec = MODELS[model_id]
                model_name = model_spec["name"]

                for seed in seeds:
                    try:
                        output = model_spec["run"](
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            config.TAUS,
                            seed,
                        )
                        quantiles = np.asarray(output["quantiles"], dtype=float)
                        point_pred = np.asarray(output["point_pred"], dtype=float)

                        expected_shape = (len(y_test), len(config.TAUS))
                        if quantiles.shape != expected_shape:
                            raise ValueError(
                                f"quantiles shape {quantiles.shape} "
                                f"!= {expected_shape}"
                            )
                        if point_pred.shape != (len(y_test),):
                            raise ValueError(
                                f"point_pred shape {point_pred.shape} "
                                f"!= {(len(y_test),)}"
                            )
                        if not np.all(np.isfinite(quantiles)):
                            raise ValueError("non-finite quantile predictions")
                        if not np.all(np.isfinite(point_pred)):
                            raise ValueError("non-finite point predictions")

                        scores = evaluate(
                            y_test,
                            quantiles,
                            config.TAUS,
                            point_pred,
                            q_true_test,
                            config.CALIBRATION_LEVEL,
                        )
                        low, high = interval_bounds(
                            quantiles, config.TAUS, config.CALIBRATION_LEVEL
                        )
                        coverage = empirical_coverage(y_test, quantiles)

                        record = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "split": split_idx,
                            "seed": seed,
                            "n_train": len(y_train),
                            "n_test": len(y_test),
                            "train_time": float(output["train_time"]),
                            "infer_time": float(output["infer_time"]),
                        }
                        record.update(scores)
                        records.append(record)

                        for tau, cov in zip(config.TAUS, coverage):
                            coverage_rows.append(
                                {
                                    "dataset": dataset_name,
                                    "model": model_name,
                                    "split": split_idx,
                                    "seed": seed,
                                    "tau": float(tau),
                                    "coverage": float(cov),
                                }
                            )

                        details.append(
                            {
                                "dataset": dataset_name,
                                "model": model_name,
                                "split": split_idx,
                                "seed": seed,
                                "y_true": y_test,
                                "point_pred": point_pred,
                                "q_low": low,
                                "q_high": high,
                            }
                        )

                        print(
                            f"   [{split_idx}/{len(splits)}] "
                            f"{model_name:<12s} seed={seed} "
                            f"pinball={scores['pinball']:.4f} "
                            f"picp={scores[f'picp_{LEVEL_TAG}']:.3f} "
                            f"cross={scores['crossing_rate']:.3f} "
                            f"({output['train_time']:.1f}s)",
                            flush=True,
                        )
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        failures.append((dataset_name, model_name, message))
                        label = f"{dataset_name}/{model_name}: {message}"
                        if label not in reported_errors:
                            print(f"   !! {label}", flush=True)
                            reported_errors.add(label)

    return records, coverage_rows, details, failures


def report(records, coverage_rows, details, failures):
    results_dir = config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(records)
    coverage_df = pd.DataFrame(coverage_rows)

    raw_path = results_dir / "metrics_raw.csv"
    coverage_path = results_dir / "calibration_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)

    present = [c for c in AGGREGATED_METRICS if c in raw_df.columns]
    grouped = raw_df.groupby(["dataset", "model"], sort=False)[present].agg(
        ["mean", "std"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index().sort_values(["dataset", "pinball_mean"])

    summary_path = results_dir / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)

    display_cols = [
        column
        for column in [
            "dataset",
            "model",
            "pinball_mean",
            "crps_mean",
            "mae_mean",
            f"picp_{LEVEL_TAG}_mean",
            f"mpiw_{LEVEL_TAG}_mean",
            "crossing_rate_mean",
            "true_q_mae_mean",
            "train_time_mean",
        ]
        if column in summary.columns
    ]
    print("\nAggregated results (mean over splits x seeds)")
    print(
        summary[display_cols].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )

    for dataset_name, group in summary.groupby("dataset", sort=False):
        best = group.loc[group["pinball_mean"].idxmin()]
        print(
            f"  Lowest mean pinball on {dataset_name}: "
            f"{best['model']} ({best['pinball_mean']:.4f})"
        )

    plot_paths = []
    try:
        import plots

        plot_paths.extend(plots.plot_calibration(coverage_df, results_dir))
        plot_paths.extend(plots.plot_crossing(raw_df, results_dir))
        plot_paths.extend(plots.plot_interval_predictions(details, results_dir))
    except Exception as exc:
        print(f"\nPlot generation failed: {type(exc).__name__}: {exc}")

    print("\nSaved outputs:")
    for path in (raw_path, summary_path, coverage_path):
        print(f"  - {path}")
    for path in plot_paths:
        print(f"  - {path}")

    if failures:
        counts = {}
        for dataset_name, model_name, message in failures:
            key = (dataset_name, model_name, message)
            counts[key] = counts.get(key, 0) + 1
        print(f"\nSkipped {len(failures)} run(s):")
        for (dataset_name, model_name, message), count in list(counts.items())[:12]:
            print(f"  - {count}x {dataset_name}/{model_name}: {message}")
        if len(counts) > 12:
            print(f"  ... and {len(counts) - 12} more distinct error(s)")


def main(argv=None):
    args = parse_args(argv)
    print("=" * 74)
    print(" Quantile Regression vs. MSE - uncertainty benchmark")
    print("=" * 74)

    interactive = sys.stdin.isatty()

    if args.datasets is not None:
        dataset_text = args.datasets
    elif interactive:
        print_menu("Datasets", DATASETS)
        dataset_text = ask("\nDatasets (e.g. 1,3 or all) [all]: ", "all")
    else:
        dataset_text = "all"
        print("\nNon-interactive stdin: using all datasets.")

    if args.models is not None:
        model_text = args.models
    elif interactive:
        print_menu("Models", MODELS)
        model_text = ask("\nModels (e.g. 2,5 or all) [all]: ", "all")
    else:
        model_text = "all"
        print("Non-interactive stdin: using all models.")

    if args.seeds is not None:
        n_seeds = args.seeds
    elif interactive:
        raw = ask(
            f"\nNumber of random seeds [{config.DEFAULT_SEEDS}] "
            "(1 = smoke test): ",
            str(config.DEFAULT_SEEDS),
        )
        try:
            n_seeds = int(raw)
        except ValueError:
            print(f"Invalid seed count: {raw!r}")
            return 2
    else:
        n_seeds = config.DEFAULT_SEEDS
        print(f"Non-interactive stdin: using {n_seeds} seeds.")

    try:
        dataset_ids = resolve_datasets(dataset_text)
        model_ids = resolve_models(model_text)
    except ValueError as exc:
        print(f"\nSelection error: {exc}")
        return 2

    if not 1 <= n_seeds <= 50:
        print("\nSeed count must be between 1 and 50.")
        return 2

    dataset_names = [DATASETS[i]["name"] for i in dataset_ids]
    model_names = [MODELS[i]["name"] for i in model_ids]
    print("\nConfiguration")
    print(f"  datasets : {', '.join(dataset_names)}")
    print(f"  models   : {', '.join(model_names)}")
    print(f"  seeds    : {n_seeds} per split")
    print(
        "  taus     : "
        + ", ".join(f"{tau:g}" for tau in config.TAUS)
    )
    print(f"  results  : {config.RESULTS_DIR}")

    records, coverage_rows, details, failures = run_benchmark(
        dataset_ids, model_ids, n_seeds
    )

    if not records:
        print("\nNo successful runs. Check the errors above.")
        return 1

    report(records, coverage_rows, details, failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
