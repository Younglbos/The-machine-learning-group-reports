"""Models receive only the current transformed batch, never labels or future data."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from models.kmeans import fit_kmeans


class StreamKMeans:
    def __init__(self, mode, clusters, seed, options, window_batches=2):
        if mode not in {"warm_start", "sliding_window", "full_retrain"}:
            raise ValueError("Unknown streaming baseline")
        self.mode, self.clusters, self.seed = mode, clusters, seed
        self.options, self.window_batches = options, window_batches
        self.history, self.model = [], None

    def update(self, features):
        previous = None if self.model is None else self.model.cluster_centers_.copy()
        if self.mode == "warm_start":
            training = features
        else:
            self.history.append(features.copy())
            if self.mode == "sliding_window":
                self.history = self.history[-self.window_batches:]
            training = np.concatenate(self.history)
        centers = previous if self.mode == "warm_start" else None
        self.model, _ = fit_kmeans(training, self.clusters, self.seed, self.options, centers)
        if previous is not None:
            # Geometric alignment uses no reference labels.
            rows, cols = linear_sum_assignment(cdist(previous, self.model.cluster_centers_))
            permutation = np.empty(self.clusters, dtype=int)
            permutation[rows] = cols
            self.model.cluster_centers_ = self.model.cluster_centers_[permutation]
        return self.predict(features)

    def predict(self, features):
        return self.model.predict(features)

    @property
    def state_bytes(self):
        centers = 0 if self.model is None else self.model.cluster_centers_.nbytes
        return centers + sum(batch.nbytes for batch in self.history)
