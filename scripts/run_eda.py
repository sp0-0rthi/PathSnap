"""Dataset understanding and EDA (report section 2.2).

Produces, from the raw dataset, everything the report's EDA section needs:
* dataset inventory (trip counts per partition, file counts, sizes)
* summary statistics (observations per trip, sampling intervals, gaps,
  missing values, trajectory extents)
* GPS trajectory plots and coordinate distributions
* road-network size/connectivity analysis (components, isolated nodes)
* an interactive Folium map of a selected training trip
* a data-quality record (issues, assumptions, decisions)

Analyses are labelled with the partition they used. Only training and
student-testing trips are explored; the evaluator partition is left
untouched until final serving.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_RAW, RESULTS
from src.data_processing import (clean_track, discover_trips, load_trip,
                                 validate_trip)
from src.road_network import RoadNetwork
from src.splits import assign_partitions

EDA_DIR = RESULTS / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    infos = discover_trips(DATA_RAW)
    if not infos:
        raise SystemExit("No trips found under data/raw - run "
                         "scripts/make_synthetic_data.py or add the dataset.")
    trip_ids = [i["trip_id"] for i in infos]
    partitions = assign_partitions(trip_ids, raw_dir=DATA_RAW)

    # ---------------- inventory ----------------
    inventory = {
        "n_trips_total": len(infos),
        "n_train": len(partitions["train"]),
        "n_student_test": len(partitions["student_test"]),
        "n_evaluator_test": len(partitions["evaluator_test"]),
        "file_counts": {},
        "partition_note": ("Dataset-provided partition layout if detected, "
                           "else deterministic seeded 80/10/10 split "
                           "(see src/splits.py)."),
    }
    for info in infos:
        for kind in ("track", "nodes", "arcs", "route"):
            key = f".{kind}"
            inventory["file_counts"][key] = inventory["file_counts"].get(key, 0) + \
                (1 if info[kind] is not None else 0)
    with open(EDA_DIR / "dataset_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

    # ---------------- per-trip quality + stats ----------------
    quality_rows, stats_rows = [], []
    allowed = set(partitions["train"]) | set(partitions["student_test"])
    for info in infos:
        if info["trip_id"] not in allowed:
            continue  # evaluator partition untouched
        trip = load_trip(info)
        part = ("train" if info["trip_id"] in partitions["train"]
                else "student_test")
        q = validate_trip(trip)
        q["trip_id"] = trip.trip_id
        q["partition"] = part
        quality_rows.append(q)

        track = clean_track(trip.track)
        ts = track["timestamp"].to_numpy(dtype=float)
        from src.geo import haversine_m
        step = haversine_m(track["lat"].iloc[:-1].to_numpy(),
                           track["lon"].iloc[:-1].to_numpy(),
                           track["lat"].iloc[1:].to_numpy(),
                           track["lon"].iloc[1:].to_numpy())
        stats_rows.append({
            "trip_id": trip.trip_id,
            "partition": part,
            "n_observations": len(track),
            "duration_s": float(ts.max() - ts.min()) if len(ts) else 0.0,
            "median_sampling_interval_s": float(np.median(np.diff(ts))) if len(ts) > 1 else np.nan,
            "path_length_m": float(step.sum()),
            "lat_min": float(track["lat"].min()), "lat_max": float(track["lat"].max()),
            "lon_min": float(track["lon"].min()), "lon_max": float(track["lon"].max()),
            "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs),
            "route_arcs": len(trip.route),
        })
    quality = pd.DataFrame(quality_rows)
    stats = pd.DataFrame(stats_rows)
    quality.to_csv(EDA_DIR / "data_quality.csv", index=False)
    stats.to_csv(EDA_DIR / "trajectory_stats.csv", index=False)

    # ---------------- summary ----------------
    summary = {
        "observations_total": int(stats["n_observations"].sum()),
        "observations_per_trip": {
            "mean": float(stats["n_observations"].mean()),
            "min": int(stats["n_observations"].min()),
            "max": int(stats["n_observations"].max()),
        },
        "median_sampling_interval_s": float(stats["median_sampling_interval_s"].median()),
        "pct_1s_steps": float(np.mean([
            np.mean(np.diff(clean_track(load_trip(i).track)["timestamp"].to_numpy()) == 1.0)
            for i in infos[:20] if i["trip_id"] in allowed])) if len(infos) else np.nan,
        "missing_gps_cells_total": int(quality["n_missing_gps_cells"].sum()),
        "trips_with_gaps_gt_5s": int((quality["n_gaps_gt_5s"] > 0).sum()),
        "max_gap_s_overall": float(quality["max_gap_s"].max()),
        "partitions_analysed": ["train", "student_test"],
    }
    with open(EDA_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------------- connectivity on a representative training trip ----
    rep_id = partitions["train"][0]
    rep = load_trip(next(i for i in infos if i["trip_id"] == rep_id))
    net = RoadNetwork(rep.nodes, rep.arcs)
    seen = np.zeros(net.n_arcs, dtype=bool)
    comps = []
    for s in range(net.n_arcs):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in net.adjacency[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    conn = {
        "representative_trip": rep_id,
        "partition": "train",
        "n_arcs": net.n_arcs,
        "n_nodes": len(rep.nodes),
        "n_connected_components": len(comps),
        "largest_component_arcs": max(len(c) for c in comps),
        "isolated_arcs": int(sum(1 for c in comps if len(c) == 1)),
        "mean_arc_length_m": float(np.mean(net.length_m)),
        "median_arc_length_m": float(np.median(net.length_m)),
    }
    with open(EDA_DIR / "network_connectivity.json", "w") as f:
        json.dump(conn, f, indent=2)

    # ---------------- plots ----------------
    # GPS coordinate distribution (train partition)
    train_ids = set(partitions["train"])
    sample = [i for i in infos if i["trip_id"] in train_ids][:40]
    fig, ax = plt.subplots(figsize=(7, 6))
    for info in sample:
        t = clean_track(load_trip(info).track)
        ax.plot(t["lon"], t["lat"], ".", ms=1, alpha=0.4)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("GPS coordinate distribution - training partition")
    fig.tight_layout(); fig.savefig(EDA_DIR / "gps_distribution_train.png", dpi=150)
    plt.close(fig)

    # trajectory plot of the representative trip
    t = clean_track(rep.track)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(t["lon"], t["lat"], ".-", ms=2, lw=0.5, color="#1f77b4")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"GPS trajectory - {rep_id} (training partition)")
    fig.tight_layout(); fig.savefig(EDA_DIR / f"trajectory_{rep_id}.png", dpi=150)
    plt.close(fig)

    # road-network visualisation of the representative trip
    # (subsampled to 20k arcs for rendering: some networks have >100k arcs)
    fig, ax = plt.subplots(figsize=(7, 6))
    arc_sel = np.linspace(0, net.n_arcs - 1, min(net.n_arcs, 20000)).astype(int)
    for i in arc_sel:
        g = net.arc_geometry_latlon(int(i))
        ax.plot([g[0][1], g[1][1]], [g[0][0], g[1][0]], color="#555555", lw=1)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Road network - {rep_id} ({net.n_arcs} arcs, training partition)")
    fig.tight_layout(); fig.savefig(EDA_DIR / f"network_{rep_id}.png", dpi=150)
    plt.close(fig)

    # NOTE: interactive Folium maps of raw GPS + network + reference +
    # prediction are produced by the main pipeline (src/visualize.py) and
    # committed under results/maps/ - see report section 6.

    # data-quality narrative record
    with open(EDA_DIR / "data_quality_notes.md", "w") as f:
        f.write(f"""# Data-quality record (partitions analysed: train + student_test)

* GPS rows inspected: {int(quality['n_gps_rows'].sum())}
* Rows with missing/unparseable cells: {int(quality['n_missing_gps_cells'].sum())} (dropped)
* Out-of-range coordinates: {int(quality['n_bad_lat'].sum() + quality['n_bad_lon'].sum())} (dropped)
* Non-monotonic timestamps: {int(quality['n_non_monotonic_ts'].sum())} (sorted; duplicates collapsed)
* Trips containing >5 s sampling gaps: {int((quality['n_gaps_gt_5s'] > 0).sum())} of {len(quality)}
* Largest single gap: {quality['max_gap_s'].max():.1f} s
* Arc rows referencing missing nodes: {int(quality['n_bad_arc_refs'].sum())}
* Route entries referencing unknown arcs: {int(quality['n_bad_route_refs'].sum())}

Assumptions recorded:
1. The vehicle traverses .route arcs in chronological order; per-observation
   reference arcs are assigned by monotonic DP alignment (see src/labeling.py).
2. Reference routes are treated as reference information, not independently
   verified ground truth (dataset brief, section 6).
3. Arcs are straight segments between their endpoint nodes.
4. Distances use haversine / a local equirectangular projection in meters.
""")
    print("EDA outputs written to", EDA_DIR)


if __name__ == "__main__":
    main()
