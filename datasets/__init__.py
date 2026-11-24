"""Dataset registry: numbered IDs and names map to data-contract loaders."""

from . import bike_sharing, california, synthetic

DATASETS = {
    "1": {
        "name": "california",
        "description": "California housing (K-Fold CV)",
        "loader": california.load,
    },
    "2": {
        "name": "bike_sharing",
        "description": "UCI Bike Sharing (chronological TimeSeriesSplit)",
        "loader": bike_sharing.load,
    },
    "3": {
        "name": "synthetic",
        "description": "Synthetic heteroscedastic (exact true quantiles)",
        "loader": synthetic.load,
    },
}

_NAME_TO_ID = {spec["name"]: key for key, spec in DATASETS.items()}


def resolve_choice(text):
    """Resolve ``'1,3'``, ``'california,synthetic'`` or ``'all'`` to dataset IDs."""
    raw = (text or "").strip().lower()
    if raw in ("", "all"):
        return list(DATASETS)

    ids = []
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        if token in DATASETS:
            key = token
        elif token in _NAME_TO_ID:
            key = _NAME_TO_ID[token]
        else:
            options = ", ".join(f"{k}={v['name']}" for k, v in DATASETS.items())
            raise ValueError(f"Unknown dataset '{token}'. Choose {options}, or 'all'.")
        if key not in ids:
            ids.append(key)
    if not ids:
        raise ValueError("No dataset selected.")
    return ids
