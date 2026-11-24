"""Shared, label-free corruptions for paired course comparisons."""

import hashlib
import numpy as np


def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask,dtype=np.uint8).tobytes()).hexdigest()


def noisy_observations(views,mask,seed,strength=0.0):
    """Noise in view 1, scaled only by its observed rows; never inspect hidden rows."""
    result=[v.copy() for v in views]
    if strength:
        observed=np.asarray(mask[:,0],dtype=bool)
        scale=views[0][observed].std(axis=0)
        rng=np.random.default_rng(seed+7000)
        result[0][observed]+=rng.normal(size=result[0][observed].shape).astype(np.float32)*scale*strength
    return result


def noisy_stream_view(raw,seed,view,drift_view,condition,strength=1.0):
    """Environment-side corruption; each learner receives exactly the same new view."""
    result=raw.copy()
    if condition=='view_noise' and view>=drift_view:
        rng=np.random.default_rng(seed+9000+view)
        result+=rng.normal(size=raw.shape).astype(np.float32)*raw.std(axis=0)*strength
    return result
