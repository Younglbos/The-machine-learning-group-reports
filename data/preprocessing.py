"""Fit per-view feature transformations only to observations permitted by a protocol."""

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize


def dense_views(views):
    return [np.asarray(v.toarray() if sparse.issparse(v) else v, dtype=np.float32) for v in views]


def validate_views(views, labels):
    if len(views) < 2 or not len(labels):
        raise ValueError("At least two nonempty aligned views are required")
    for view in views:
        if view.ndim != 2 or view.shape[0] != len(labels) or not np.isfinite(view).all():
            raise ValueError("Invalid features or inconsistent sample counts")


def prepare_observed(views, mask, fill="mean"):
    """Return scaled filled views and fitted scalers; unobserved values are never read."""
    if fill not in {"zero", "mean"}:
        raise ValueError("Unknown fill method")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(views[0]), len(views)) or not mask.any(axis=1).all():
        raise ValueError("Invalid sample-view mask")
    blocks, scalers = [], []
    for j, raw in enumerate(views):
        observed = mask[:, j]
        if observed.sum() < 2:
            raise ValueError("Each view must retain at least two observations")
        scaler = MinMaxScaler()
        known = scaler.fit_transform(raw[observed])
        block = np.zeros(raw.shape, dtype=np.float32)
        if fill == "mean":
            block[:] = known.mean(axis=0)
        block[observed] = known
        blocks.append(block)
        scalers.append(scaler)
    return blocks, scalers


class FrozenStreamEncoder:
    """First-batch MinMax/PCA, then frozen coordinates for all subsequent batches."""
    def __init__(self, components=16, seed=1):
        self.components, self.seed = components, seed

    def fit(self, views):
        self.scalers, self.projections = [], []
        for raw in views:
            scaler = MinMaxScaler().fit(raw)
            known = scaler.transform(raw)
            rank = min(self.components, len(raw) - 1, raw.shape[1])
            projection = PCA(n_components=rank, svd_solver="randomized", random_state=self.seed).fit(known)
            self.scalers.append(scaler)
            self.projections.append(projection)
        return self

    def transform(self, views):
        return np.concatenate([normalize(projection.transform(scaler.transform(raw)))
            for raw, scaler, projection in zip(views, self.scalers, self.projections)], axis=1).astype(np.float32)
