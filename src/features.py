"""Feature engineering.

One row per (observation, candidate arc) with the features:

    dist_to_segment : distance (m) from the GPS point to the candidate
                      arc's geometry (true point-to-segment distance, not
                      the midpoint distance). Closeness is the strongest
                      signal that an arc produced the observation.
    heading_diff    : smallest angle (degrees, [0, 90]) between the GPS
                      movement heading and the arc's bearing, taken over
                      both arc directions because arcs are traversed in
                      either direction. Vehicle motion must agree with
                      the road direction.
    speed           : GPS speed (m/s) at the observation, from consecutive
                      positions over the timestamp delta. Repeated per
                      candidate of the same observation; it lets the model
                      use motion plausibility (e.g. stationary points hug
                      the road they are on).

GPS heading uses a central difference over neighbours to smooth noise;
across a timestamp gap the difference is one-sided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import angle_diff_deg, bearing_deg, point_to_segment_m
from .road_network import RoadNetwork


def observation_kinematics(track: pd.DataFrame, network: RoadNetwork):
    """Heading and speed per observation from the raw GPS track."""
    lat = track["lat"].to_numpy(dtype=float)
    lon = track["lon"].to_numpy(dtype=float)
    ts = track["timestamp"].to_numpy(dtype=float)
    x, y = network.proj.to_xy(lat, lon)
    n = len(track)

    heading = np.full(n, np.nan)
    speed = np.zeros(n)
    if n >= 2:
        dx_prev, dy_prev = np.diff(x), np.diff(y)
        dt = np.diff(ts)
        dt_safe = np.where(dt > 0, dt, np.nan)
        seg_speed = np.hypot(dx_prev, dy_prev) / dt_safe
        seg_head = bearing_deg(x[:-1], y[:-1], x[1:], y[1:])
        gap = (dt > 5.0)  # one-sided differences across gaps
        # central difference where both neighbours are usable
        for i in range(n):
            if i == 0:
                heading[i] = seg_head[0]
                speed[i] = seg_speed[0]
            elif i == n - 1:
                heading[i] = seg_head[-1]
                speed[i] = seg_speed[-1]
            elif gap[i - 1] or gap[i]:
                heading[i] = seg_head[i] if not gap[i] else seg_head[i - 1]
                speed[i] = seg_speed[i] if not gap[i] else seg_speed[i - 1]
            else:
                heading[i] = bearing_deg(x[i - 1], y[i - 1], x[i + 1], y[i + 1])
                speed[i] = np.hypot(x[i + 1] - x[i - 1], y[i + 1] - y[i - 1]) / (ts[i + 1] - ts[i - 1])
    speed = np.nan_to_num(speed, nan=0.0)
    heading = np.nan_to_num(heading, nan=0.0)
    return heading, speed


def build_feature_rows(track: pd.DataFrame, network: RoadNetwork,
                       cand_lists, ref_arc_idx=None) -> pd.DataFrame:
    """Build the candidate-level feature table for one trip.

    ref_arc_idx: internal arc indices of the reference arc per observation
    (-1 where unknown). When supplied, the binary label `is_match` marks
    the candidate equal to the reference arc.
    """
    lat = track["lat"].to_numpy(dtype=float)
    lon = track["lon"].to_numpy(dtype=float)
    heading, speed = observation_kinematics(track, network)
    px, py = network.proj.to_xy(lat, lon)

    # Flatten candidate lists into arrays, then compute every feature
    # vectorised over all (observation, candidate) rows at once.
    obs_idx = np.concatenate(
        [np.full(len(c), i, dtype=int) for i, c in enumerate(cand_lists)])
    arc_idx = np.concatenate(
        [np.asarray(c, dtype=int) for c in cand_lists])
    a0 = network.p0[arc_idx]          # (rows, 2)
    a1 = network.p1[arc_idx]
    dist = point_to_segment_m(px[obs_idx], py[obs_idx],
                              a0[:, 0], a0[:, 1], a1[:, 0], a1[:, 1])
    seg_bearing = network.bearing[arc_idx]
    h_obs = heading[obs_idx]
    h_diff = np.minimum(angle_diff_deg(h_obs, seg_bearing),
                        angle_diff_deg(h_obs, (seg_bearing + 180.0) % 360.0))
    data = {
        "obs_idx": obs_idx,
        "arc_idx": arc_idx,
        "dist_to_segment": dist,
        "heading_diff": h_diff,
        "speed": speed[obs_idx],
    }
    if ref_arc_idx is not None:
        data["is_match"] = (np.asarray(ref_arc_idx)[obs_idx] == arc_idx).astype(int)
    return pd.DataFrame(data)
