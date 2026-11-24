"""Permutation-invariant clustering evaluation. Labels never enter model fitting."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def clustering_accuracy(truth, predicted):
    truth, predicted = np.asarray(truth).ravel(), np.asarray(predicted).ravel()
    if not len(truth) or truth.shape != predicted.shape:
        raise ValueError("Labels must be nonempty vectors of equal length")
    _, a = np.unique(truth, return_inverse=True)
    _, b = np.unique(predicted, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / len(truth))


def scores(truth, predicted):
    acc = clustering_accuracy(truth, predicted)
    return {"acc": acc, "nmi": float(normalized_mutual_info_score(truth, predicted)),
            "ari": float(adjusted_rand_score(truth, predicted))}


def silhouette(features, predicted, seed, sample_size=256):
    unique = len(np.unique(predicted))
    if not 1 < unique < len(predicted):
        return None
    try:
        return float(silhouette_score(features, predicted, sample_size=min(sample_size, len(predicted)),
                                      random_state=seed))
    except ValueError:
        return None
