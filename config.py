"""Central configuration for the quantile-regression vs. MSE benchmark."""

import os
from pathlib import Path

import numpy as np

# Tau grid used for every quantile model and every metric.
TAUS = np.round(np.arange(0.1, 1.0, 0.1), 1)

DEFAULT_SEEDS = 5
SEED_START = 0

# Dataset split sizes.
N_FOLDS = 5            # California housing K-Fold
BIKE_SPLITS = 3        # chronological cuts for bike sharing
SYNTH_SPLITS = 3       # independent synthetic draws
SYNTH_SAMPLES = 1800
VAL_FRACTION = 0.2     # validation slice carved out of each training pool

# Metrics: PICP / MPIW are reported for the central 80% interval.
CALIBRATION_LEVEL = 0.8

# Neural network training.
NN_HIDDEN = 64
NN_EPOCHS = 200
NN_PATIENCE = 15
NN_LR = 1e-3
NN_BATCH = 256

# Output.
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "datasets" / "data"
PLOT_DPI = 130
MAX_PLOT_POINTS = 120

# Model code is developed outside this repository. The default sibling folder
# must contain a ``models`` package; override it with BENCH_MODELS_ROOT.
MODELS_ROOT = Path(os.environ.get("BENCH_MODELS_ROOT", ROOT.parent / "PROJETO_models"))
