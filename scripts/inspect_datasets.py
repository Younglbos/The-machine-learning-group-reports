"""Check dimensions, labels and finite values of the downloaded datasets."""

import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.load_dataset import load_dataset


def main():
    entries = json.loads((ROOT / "data" / "dataset_sources.json").read_text(encoding="utf-8"))
    records = []
    for entry in entries:
        views, labels = load_dataset(entry["name"])
        if not len(labels) or not np.isfinite(labels).all() or len(views) < 2:
            raise ValueError(f"Invalid views or labels: {entry['name']}")
        details = []
        for i, view in enumerate(views):
            if len(view.shape) != 2 or view.shape[0] != len(labels):
                raise ValueError(f"Sample count mismatch: {entry['name']}, view {i + 1}")
            values = view.data if sparse.issparse(view) else view
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite features: {entry['name']}, view {i + 1}")
            zero_rows = int(np.count_nonzero(np.asarray(abs(view).sum(axis=1)).ravel() == 0))
            details.append({"view": i + 1, "shape": list(view.shape), "dtype": str(view.dtype),
                            "sparse": bool(sparse.issparse(view)), "zero_rows": zero_rows})
        classes, counts = np.unique(labels, return_counts=True)
        record = {
            "name": entry["name"], "samples": len(labels), "views": len(views),
            "classes": len(classes), "feature_dimensions": [v.shape[1] for v in views],
            "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
            "view_details": details, "all_features_finite": True,
            "sample_counts_match": True, "source_url": entry["source_url"],
        }
        records.append(record)
        print(f"{record['name']}: N={record['samples']}, V={record['views']}, "
              f"K={record['classes']}, dims={record['feature_dimensions']}")
    (ROOT / "data" / "dataset_summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All {len(records)} main datasets passed structural and finite-value checks.")


if __name__ == "__main__":
    main()
