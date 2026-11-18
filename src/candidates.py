# PathSnap v3.1 - data pipeline (owner: Spoorthi)
"""Candidate generation.

For each GPS observation we take the k road arcs whose midpoints are
nearest to the observation (BallTree/KDTree over arc midpoints) within a
search radius of ~50 m.

Edge case - missing candidates: if no arc midpoint falls inside the
radius, the query is retried once with a relaxed radius; if that also
fails the observation is left with an empty candidate set, which the
classifier (nearest-arc fallback) and the Viterbi decoder (state skip)
handle explicitly.

Candidate Miss Rate: the fraction of observations whose reference arc is
absent from the generated candidate set. This is a property of the
candidate generator, not of any model, and bounds the best achievable
Top-1 accuracy.
"""

from __future__ import annotations

import numpy as np

from .config import CFG
from .road_network import RoadNetwork


def generate_candidates(lats, lons, network: RoadNetwork,
                        k: int = CFG.candidate_k,
                        radius_m: float = CFG.candidate_radius_m,
                        relaxed_radius_m: float = CFG.candidate_relaxed_radius_m):
    """Generate candidate arcs for a sequence of GPS observations.

    Returns
    -------
    cand_lists : list per observation of arc indices (internal indices).
    relaxed    : bool array, True where the relaxed radius was needed.
    """
    cand = network.query_candidates_exact(lats, lons, k=k, radius_m=radius_m)
    relaxed = np.zeros(len(cand), dtype=bool)
    missing = [i for i, c in enumerate(cand) if not c]
    if missing:
        lat_arr = np.asarray(lats, dtype=float)
        lon_arr = np.asarray(lons, dtype=float)
        retry = network.query_candidates_exact(lat_arr[missing], lon_arr[missing],
                                               k=k, radius_m=relaxed_radius_m)
        for i, c in zip(missing, retry):
            cand[i] = c
            relaxed[i] = True
    return cand, relaxed


def candidate_miss_rate(cand_lists, ref_arc_idx) -> float:
    """Fraction of observations whose reference arc is not a candidate."""
    ref = np.asarray(ref_arc_idx)
    valid = ref >= 0
    if valid.sum() == 0:
        return float("nan")
    misses = sum(1 for c, r in zip(np.asarray(cand_lists, dtype=object)[valid], ref[valid])
                 if r not in c)
    return misses / float(valid.sum())
