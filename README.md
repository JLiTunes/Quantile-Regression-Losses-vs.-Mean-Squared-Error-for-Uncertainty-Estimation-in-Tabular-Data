# Quantile Regression vs. MSE for Uncertainty Estimation

A lightweight benchmark for comparing quantile regression losses with
MSE-based uncertainty estimation on tabular data.

The harness trains every model on the same leakage-safe splits and reports
pinball loss, CRPS, calibration, interval width, quantile crossings, and
point-prediction errors.

## Datasets

| ID | Dataset | Splits |
|----|---------|--------|
| 1 | California housing | 5-fold cross-validation |
| 2 | Bike sharing | 3 chronological splits |
| 3 | Synthetic heteroscedastic | 3 independent draws |

Bike-sharing data is downloaded automatically on first use. A `StandardScaler`
is fit only on each training split.

## Models

Model implementations are kept outside this repository. The harness loads them
from a `models` package inside `BENCH_MODELS_ROOT` (default:
`../PROJETO_models`):

```bash
export BENCH_MODELS_ROOT=/path/to/PROJETO_models
```

The external `models/__init__.py` must expose `MODELS` and `resolve_choice()`.
Each model provides:

```python
run_model(X_train, y_train, X_val, y_val, X_test, taus, seed)
```

and returns:

```python
{
    "quantiles": np.ndarray,  # (n_test, n_taus)
    "point_pred": np.ndarray, # (n_test,)
    "train_time": float,
    "infer_time": float,
}
```

Current models: `mse_homo`, `mse_hetero`, `linear_qr`, `qrf`, `gbdt_qr`,
`nn_qr_raw`, and `nn_qr_mono`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the interactive menu:

```bash
python main.py
```

Or select datasets, models, and seeds directly:

```bash
python main.py --datasets all --models all --seeds 1
python main.py --datasets synthetic --models nn_qr_raw,nn_qr_mono --seeds 1
```

Dataset IDs: `1` california, `2` bike_sharing, `3` synthetic.
Model IDs: `1` mse_homo, `2` mse_hetero, `3` linear_qr, `4` qrf,
`5` gbdt_qr, `6` nn_qr_raw, `7` nn_qr_mono.

## Outputs

Results are written to `results/`:

- `metrics_raw.csv` — one row per dataset, split, model, and seed
- `metrics_summary.csv` — mean/std aggregated by dataset and model
- `calibration_raw.csv` — empirical coverage per quantile level
- `calibration_*.png`, `intervals_*.png`, `crossing_rate.png` — plots

## Metrics

Pinball loss, CRPS, MAE, RMSE, PICP and MPIW for the central 80% interval,
quantile crossing rate, and true-quantile MAE (synthetic data only).

## Structure

```
main.py            CLI, orchestration, result aggregation
config.py          Quantile grid, seeds, split sizes, model path
metrics.py         Pinball, CRPS, coverage, width, crossings
plots.py           Calibration, interval, and crossing plots
datasets/          California, bike sharing, and synthetic loaders
requirements.txt   Python dependencies
```
