"""Read original multi-view features, retaining sample order and label values."""

import json
from pathlib import Path
from scipy.io import loadmat

DATA_DIR = Path(__file__).resolve().parent


def load_dataset(name):
    """Return (list of N-by-D view matrices, 1D evaluation labels).

    Accept a manifest filename or stem, e.g. BBCSport.mat or BBCSport.
    No normalization, imputation, label remapping or shuffling is applied.
    """
    entries = json.loads((DATA_DIR / "dataset_sources.json").read_text(encoding="utf-8"))
    entry = next((e for e in entries if name in (e["name"], Path(e["name"]).stem)), None)
    if entry is None:
        raise ValueError(f"Unknown dataset: {name}")
    view_key, label_key = entry["view_key"], entry["label_key"]
    # Explicit selection bypasses the BBC4view mirror's trailing bytes.
    contents = loadmat(DATA_DIR / "raw" / entry["name"], variable_names=[view_key, label_key])
    return list(contents[view_key].ravel()), contents[label_key].ravel()
