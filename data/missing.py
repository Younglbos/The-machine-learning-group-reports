"""Whole sample-view dropout with an explicitly defined incomplete-sample fraction."""

import numpy as np


def make_mask(samples, views, rate, seed, pattern="random", labels=None):
    """Select floor(N*rate) samples and remove exactly one entire view per sample.

    Every object retains >=1 view. Actual block dropout is floor(N*rate)/(N*V).
    Labels are permitted only in the synthetic class-related missingness generator.
    """
    if views < 2 or samples < 2 or not 0 <= rate <= 1:
        raise ValueError("Invalid mask dimensions or rate")
    if pattern not in {"random", "single_view", "class_related"}:
        raise ValueError("Unknown missingness pattern")
    rng = np.random.default_rng(seed)
    mask = np.ones((samples, views), dtype=bool)
    if pattern == "class_related":
        if labels is None or len(labels) != samples:
            raise ValueError("Synthetic class-related missingness needs labels")
        _, category = np.unique(labels, return_inverse=True)
        weights = 1 + 3 * category / max(1, category.max())
        order = np.argsort(-np.log(np.clip(rng.random(samples), 1e-12, 1)) / weights)
        lost = category % views
    else:
        order = rng.permutation(samples)
        lost = rng.integers(views, size=samples) if pattern == "random" else np.zeros(samples, dtype=int)
    selected = order[:int(samples * rate)]
    mask[selected, lost[selected]] = False
    return mask


def mask_summary(mask):
    return {"incomplete_sample_fraction": float((~mask.all(axis=1)).mean()),
            "missing_block_fraction": float((~mask).mean()),
            "fully_unobserved_samples": int((~mask.any(axis=1)).sum()),
            "observed_per_view": mask.sum(axis=0).tolist()}
