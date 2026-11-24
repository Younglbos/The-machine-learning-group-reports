"""Offline-only comparison of true, completed and consensus neighborhood structure."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def neighbors(features, k=10):
    k = min(k, len(features) - 1)
    found = NearestNeighbors(n_neighbors=k+1, algorithm='brute').fit(features).kneighbors(features, return_distance=False)
    return np.stack([row[row != i][:k] for i,row in enumerate(found)])


def overlap(reference, candidate, selected=None):
    indices = np.arange(len(reference)) if selected is None else np.flatnonzero(selected)
    if not len(indices):
        return None
    return float(np.mean([len(set(reference[i]) & set(candidate[i])) / reference.shape[1] for i in indices]))


def diagnose(true_latents, completed, consensus, mask, truth):
    consensus_neighbors = neighbors(consensus)
    rows = []
    for j, (original, filled) in enumerate(zip(true_latents, completed)):
        original_neighbors, filled_neighbors = neighbors(original), neighbors(filled)
        missing = ~mask[:,j]
        row = {'view': j+1, 'missing_samples': int(missing.sum()), 'k': original_neighbors.shape[1],
               'completion_overlap_all': overlap(original_neighbors, filled_neighbors),
               'completion_overlap_missing': overlap(original_neighbors, filled_neighbors, missing),
               'completion_overlap_observed': overlap(original_neighbors, filled_neighbors, ~missing),
               'consensus_overlap_with_true_view': overlap(original_neighbors, consensus_neighbors),
               'true_neighbor_label_purity': float((truth[original_neighbors] == truth[:,None]).mean()),
               'completed_neighbor_label_purity': float((truth[filled_neighbors] == truth[:,None]).mean())}
        rows.append(row)
    return rows
