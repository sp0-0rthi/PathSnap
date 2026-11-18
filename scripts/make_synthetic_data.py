"""Generate a synthetic stand-in dataset in EXACTLY the raw format.

Layout per trip directory (identical to the real dataset):
    <trip_id>/<trip_id>.track   lon lat timestamp   (whitespace, no header)
    <trip_id>/<trip_id>.nodes   lon lat             (row index = node id)
    <trip_id>/<trip_id>.arcs    from_node to_node   (row index = arc id)
    <trip_id>/<trip_id>.route   arc_id per line     (chronological)

This exists so the full pipeline can be developed and smoke-tested before
(or after) the real dataset arrives - the loaders treat both identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

M_PER_DEG_LAT = 111_320.0


def build_grid(n_side: int, spacing_m: float, lat0: float, lon0: float):
    """Square street grid; returns nodes (lat, lon) and arcs (i, j)."""
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(lat0))
    nodes = []
    for r in range(n_side):
        for c in range(n_side):
            lat = lat0 + r * spacing_m / M_PER_DEG_LAT
            lon = lon0 + c * spacing_m / m_per_deg_lon
            nodes.append((lon, lat))
    arcs = []
    for r in range(n_side):
        for c in range(n_side):
            i = r * n_side + c
            if c + 1 < n_side:
                arcs.append((i, i + 1))
            if r + 1 < n_side:
                arcs.append((i, i + n_side))
    return nodes, arcs


def simulate_trip(rng, nodes, arcs, n_seconds: int, noise_m: float,
                  lat0: float, gap: bool):
    """Random drive over the grid with noisy 1 Hz GPS sampling."""
    edge_of = {}
    neighbours: dict[int, list[int]] = {}
    for arc_id, (u, v) in enumerate(arcs):
        edge_of[frozenset((u, v))] = arc_id
        neighbours.setdefault(u, []).append(v)
        neighbours.setdefault(v, []).append(u)

    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(lat0))
    speed = rng.uniform(6, 12)          # m/s
    step_m = speed * 1.0                # 1 Hz

    cur = rng.integers(0, len(nodes))
    route: list[int] = []
    points = []  # (lon, lat, t)
    t = 0.0
    # position starts at node `cur`
    pos = np.array(nodes[cur])
    prev_node = cur
    dist_on_edge = 0.0

    while len(points) < n_seconds:
        nxt_options = neighbours.get(cur, [])
        if not nxt_options:
            break
        nxt = int(rng.choice(nxt_options))
        arc_id = edge_of[frozenset((cur, nxt))]
        if not route or route[-1] != arc_id:
            route.append(arc_id)
        a = np.array(nodes[cur])
        b = np.array(nodes[nxt])
        edge_len_m = np.hypot((b[0] - a[0]) * m_per_deg_lon,
                              (b[1] - a[1]) * M_PER_DEG_LAT)
        # walk along this edge in 1 s steps
        travelled = 0.0
        while travelled < edge_len_m and len(points) < n_seconds:
            frac = travelled / edge_len_m
            p = a + frac * (b - a)
            noisy = p + rng.normal(0, noise_m, size=2) / np.array(
                [m_per_deg_lon, M_PER_DEG_LAT])
            points.append((noisy[0], noisy[1], t))
            t += 1.0
            travelled += step_m
            # occasional GPS dropout: jump time forward (a gap)
            if gap and rng.random() < 0.004:
                t += float(rng.integers(30, 120))
        cur = nxt
        prev_node = cur
        dist_on_edge = 0.0

    return points, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--trips", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=int, default=15)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Bengaluru-ish reference coordinate (matches the user region; any
    # plausible city centre works for the stand-in)
    lat0, lon0 = 12.9350, 77.6050
    nodes, arcs = build_grid(args.grid, spacing_m=100.0, lat0=lat0, lon0=lon0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(args.trips):
        tid = f"trip_{k:03d}"
        d = out / tid
        d.mkdir(parents=True, exist_ok=True)
        n_seconds = int(rng.integers(180, 600))
        has_gap = rng.random() < 0.3
        pts, route = simulate_trip(rng, nodes, arcs, n_seconds,
                                   noise_m=12.0, lat0=lat0, gap=has_gap)
        with open(d / f"{tid}.track", "w") as f:
            for lon, lat, t in pts:
                f.write(f"{lon:.7f} {lat:.7f} {t:.1f}\n")
        with open(d / f"{tid}.nodes", "w") as f:
            for lon, lat in nodes:
                f.write(f"{lon:.7f} {lat:.7f}\n")
        with open(d / f"{tid}.arcs", "w") as f:
            for u, v in arcs:
                f.write(f"{u} {v}\n")
        with open(d / f"{tid}.route", "w") as f:
            for a in route:
                f.write(f"{a}\n")
    print(f"Wrote {args.trips} synthetic trips to {out.resolve()}")


if __name__ == "__main__":
    main()
