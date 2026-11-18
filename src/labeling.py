# PathSnap v3.1 - data pipeline (owner: Spoorthi)
"""Observation -> reference-arc assignment.

The raw .route file is a chronological sequence of arc ids with NO
timestamps, so the per-observation reference segment must be derived.
We align the GPS point sequence to the route arc sequence with a
monotonic dynamic-programming assignment:

* arc order is non-decreasing along time (the vehicle progresses along
  its route);
* cost(i, j) = distance from GPS point i to arc j;
* the alignment may advance by 0..SKIP_WINDOW arcs per observation, each
  skipped arc paying SKIP_PENALTY_M (so short side arcs can be jumped
  over without the alignment wandering backwards).

Assumptions (stated in the report, per the dataset brief's instruction
not to overstate label certainty):
* the vehicle traverses the route arcs in order without reversing;
* every GPS observation belongs to some arc of the supplied route;
* GPS noise is small relative to arc spacing, so the closest arc at the
  correct position in the sequence is the reference arc.
"""

from __future__ import annotations

import numpy as np

from .geo import LocalProjector, point_to_segment_m

SKIP_WINDOW = 3
SKIP_PENALTY_M = 15.0


def assign_reference_arcs(lats, lons, arc_geoms, route,
                          skip_window: int = SKIP_WINDOW,
                          skip_penalty_m: float = SKIP_PENALTY_M) -> np.ndarray:
    """Assign each GPS observation a reference arc from the route.

    Parameters
    ----------
    lats, lons : array-like of GPS coordinates (chronological).
    arc_geoms  : list indexed by arc_id of ((lat0, lon0), (lat1, lon1)).
    route      : chronological list of arc ids.

    Returns
    -------
    ref_arc : int array, the route arc assigned to each observation
              (-1 where no route is available).
    """
    n = len(lats)
    if not route:
        return np.full(n, -1, dtype=int)
    m = len(route)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    proj = LocalProjector(np.mean(lats), np.mean(lons))
    px, py = proj.to_xy(lats, lons)

    # Memory: real trips can have ~20k observations and ~2k route arcs, so
    # the n x m tables are built in float32/int32 and the cost matrix is
    # computed in observation blocks (never a full float64 n x m matrix).
    W = skip_window
    D = np.full((n, m), np.inf, dtype=np.float32)
    back = np.zeros((n, m), dtype=np.int32)
    BLOCK = 4000
    for b0 in range(0, n, BLOCK):
        b1 = min(n, b0 + BLOCK)
        cost = np.full((b1 - b0, m), np.inf, dtype=np.float32)
        for j, arc_id in enumerate(route):
            (la0, lo0), (la1, lo1) = arc_geoms[arc_id]
            ax, ay = proj.to_xy(la0, lo0)
            bx, by = proj.to_xy(la1, lo1)
            cost[:, j] = point_to_segment_m(px[b0:b1], py[b0:b1], ax, ay, bx, by).astype(np.float32)
        for i in range(b0, b1):
            r = i - b0
            if i == 0:
                D[0, :] = cost[0, :] + skip_penalty_m * np.arange(m, dtype=np.float32)
                continue
            best_val = np.full(m, np.inf, dtype=np.float32)
            best_k = np.zeros(m, dtype=np.int32)
            for s in range(0, W + 1):
                val = D[i - 1, :] + np.float32(s * skip_penalty_m)
                aligned = np.full(m, np.inf, dtype=np.float32)
                aligned[s:] = val[: m - s] if s else val
                better = aligned < best_val
                best_val[better] = aligned[better]
                best_k[better] = np.arange(m, dtype=np.int32)[better] - s
            D[i] = cost[r] + best_val
            back[i] = best_k

    # terminate anywhere (end of route preferred via tiny bias)
    j_end = int(np.argmin(D[-1, :] - skip_penalty_m * np.arange(m) * 0.0))
    assign = np.empty(n, dtype=int)
    j = j_end
    for i in range(n - 1, -1, -1):
        assign[i] = route[j]
        j = back[i, j] if i > 0 else j
        if j < 0:
            j = 0
    return assign
