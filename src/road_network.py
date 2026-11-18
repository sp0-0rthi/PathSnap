"""Per-trip road network: geometry, spatial index and connectivity.

Each trip ships its own .nodes/.arcs files, so a RoadNetwork is built per
trip. Arcs are straight segments between their endpoint nodes.

Provides:
* arc geometries, lengths, bearings and midpoints;
* a BallTree (haversine) over arc midpoints for candidate generation;
* a connectivity graph over arcs (arcs sharing an endpoint node, plus
  endpoints snapped within a small tolerance for connectivity gaps);
* bounded Dijkstra path distances between arcs, used by the HMM
  transition model.
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .geo import EARTH_R_M, LocalProjector, bearing_deg


class RoadNetwork:
    """Road network for one trip."""

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame,
                 snap_tolerance_m: float = 5.0):
        nodes = nodes.sort_values("node_id").reset_index(drop=True)
        arcs = arcs.sort_values("arc_id").reset_index(drop=True)
        self.arc_ids = arcs["arc_id"].to_numpy(dtype=int)
        self.n_arcs = len(arcs)

        lon = nodes["lon"].to_numpy(dtype=float)
        lat = nodes["lat"].to_numpy(dtype=float)
        self.proj = LocalProjector(float(np.mean(lat)), float(np.mean(lon)))
        nx, ny = self.proj.to_xy(lat, lon)

        fr = arcs["from_node"].to_numpy(dtype=int)
        to = arcs["to_node"].to_numpy(dtype=int)
        self.from_node, self.to_node = fr, to
        self.p0 = np.column_stack([nx[fr], ny[fr]])          # (x, y) meters
        self.p1 = np.column_stack([nx[to], ny[to]])
        self.latlon0 = np.column_stack([lat[fr], lon[fr]])
        self.latlon1 = np.column_stack([lat[to], lon[to]])
        self.mid = (self.p0 + self.p1) / 2.0
        self.mid_lat, self.mid_lon = self.proj.to_latlon(self.mid[:, 0], self.mid[:, 1])
        self.length_m = np.hypot(self.p1[:, 0] - self.p0[:, 0],
                                 self.p1[:, 1] - self.p0[:, 1])
        self.bearing = bearing_deg(self.p0[:, 0], self.p0[:, 1],
                                   self.p1[:, 0], self.p1[:, 1])

        # arc_id (as supplied) -> positional index used internally
        self.id_to_idx = {int(a): i for i, a in enumerate(self.arc_ids)}

        # BallTree over midpoints, haversine on radians
        mid_rad = np.radians(np.column_stack([self.mid_lat, self.mid_lon]))
        self._tree = BallTree(mid_rad, metric="haversine")

        # Connectivity: arcs sharing a node, plus snapped endpoints
        self.adjacency: list[set[int]] = [set() for _ in range(self.n_arcs)]
        endpoint_owner: dict[int, list[int]] = {}
        for i in range(self.n_arcs):
            endpoint_owner.setdefault(fr[i], []).append(i)
            endpoint_owner.setdefault(to[i], []).append(i)
        for owner_arcs in endpoint_owner.values():
            for a in owner_arcs:
                self.adjacency[a].update(x for x in owner_arcs if x != a)
        self._snap_endpoints(snap_tolerance_m)

    # ------------------------------------------------------------------
    def _snap_endpoints(self, tol_m: float):
        """Connect arc endpoints that are within tol_m without sharing a
        node id (absorbs small digitisation gaps in the network)."""
        endpoints = np.vstack([self.p0, self.p1])
        owner = np.array([[i, 0] for i in range(self.n_arcs)]
                         + [[i, 1] for i in range(self.n_arcs)])
        # grid bucketing keeps this near-linear for large networks
        cell = max(tol_m, 1.0)
        buckets: dict[tuple, list[int]] = {}
        keys = np.floor(endpoints / cell).astype(int)
        for idx, k in enumerate(keys):
            buckets.setdefault((k[0], k[1]), []).append(idx)
        for (kx, ky), members in buckets.items():
            neigh = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neigh.extend(buckets.get((kx + dx, ky + dy), []))
            for i in members:
                ai, _ = owner[i]
                for j in neigh:
                    if j <= i:
                        continue
                    aj, _ = owner[j]
                    if ai == aj:
                        continue
                    if np.hypot(*(endpoints[i] - endpoints[j])) <= tol_m:
                        self.adjacency[ai].add(aj)
                        self.adjacency[aj].add(ai)

    # ------------------------------------------------------------------
    def query_candidates(self, lat, lon, k: int, radius_m: float):
        """k nearest arc midpoints within radius_m of each query point.

        Returns per point a list of arc indices. Points with no arc inside
        the radius get an empty list (handled downstream as the missing-
        candidate edge case).
        """
        q = np.radians(np.column_stack([np.atleast_1d(lat), np.atleast_1d(lon)]))
        radius_rad = radius_m / EARTH_R_M
        k = min(k, self.n_arcs)
        dist, ind = self._tree.query(q, k=k)
        dist_m = dist * EARTH_R_M
        out = []
        for row_d, row_i in zip(dist_m, ind):
            keep = row_d <= radius_m
            out.append(row_i[keep].tolist())
        return out

    def query_candidates_exact(self, lat, lon, k: int, radius_m: float,
                               prefilter: int = 160):
        """k nearest arcs by EXACT point-to-segment distance within radius_m.

        The BallTree over arc midpoints only prefilters (long arcs have
        midpoints far from points right next to the arc, which inflated
        the Candidate Miss Rate). Candidates are then ranked by the true
        distance from the observation to the segment.
        """
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        q = np.radians(np.column_stack([lat, lon]))
        m = min(self.n_arcs, max(prefilter, 4 * k))
        _, ind = self._tree.query(q, k=m)
        px, py = self.proj.to_xy(lat, lon)
        out = []
        for x, y, idx in zip(px, py, ind):
            ax = self.p0[idx, 0]; ay = self.p0[idx, 1]
            dx = self.p1[idx, 0] - ax; dy = self.p1[idx, 1] - ay
            l2 = dx * dx + dy * dy
            l2[l2 == 0.0] = 1e-9
            t = np.clip(((x - ax) * dx + (y - ay) * dy) / l2, 0.0, 1.0)
            d = np.hypot(x - (ax + t * dx), y - (ay + t * dy))
            keep = d <= radius_m
            order = np.argsort(np.where(keep, d, np.inf))[:k]
            out.append([int(idx[o]) for o in order if keep[o]])
        return out

    def path_distance(self, a: int, b: int, cutoff_m: float):
        """Shortest path distance (m) between arcs a and b through the
        connectivity graph, or None if unreachable within cutoff_m.

        Cost model: travelling from arc a to adjacent arc b costs half of
        each arc's length (midpoint to midpoint), which approximates the
        network distance between points anywhere on the arcs.
        """
        if a == b:
            return 0.0
        dist = {a: 0.0}
        heap = [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            if d > cutoff_m:
                break
            for v in self.adjacency[u]:
                w = 0.5 * (self.length_m[u] + self.length_m[v])
                nd = d + w
                if nd < dist.get(v, np.inf) and nd <= cutoff_m:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist.get(b)

    def arc_geometry_latlon(self, arc_idx: int):
        """[[lat0, lon0], [lat1, lon1]] for an internal arc index."""
        return [self.latlon0[arc_idx].tolist(), self.latlon1[arc_idx].tolist()]
