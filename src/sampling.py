"""Training-row subsampling.

GPS is sampled at ~1 Hz, so consecutive observations are highly
autocorrelated: neighbouring rows carry nearly identical information.
For MODEL FITTING ONLY we keep every Nth observation per trip (stride
sampling, deterministic) capped at a configurable number per trip. This
keeps cross-validation and HPO computationally feasible on long trips
without changing the candidate distribution.

All EVALUATION always uses every observation of the student-test trips;
subsampling never touches evaluation, metrics, or the evaluator
partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CFG


def subsample_for_training(df: pd.DataFrame,
                           max_obs_per_trip: int = CFG.train_max_obs_per_trip
                           ) -> pd.DataFrame:
    """Keep a deterministic, evenly spaced subset of observations per trip."""
    parts = []
    for _, g in df.groupby("trip_id"):
        n_obs = int(g["obs_idx"].max()) + 1
        stride = max(1, int(np.ceil(n_obs / max_obs_per_trip)))
        parts.append(g[g["obs_idx"] % stride == 0])
    return pd.concat(parts, ignore_index=True)
