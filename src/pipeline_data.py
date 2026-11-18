# PathSnap v3.1 - data pipeline (owner: Spoorthi)
"""Build the candidate-level modelling dataset from raw trips.

For every trip: load -> clean -> build road network -> assign reference
arcs along the .route -> generate candidates -> compute features. The
result is cached to data/processed/features.parquet together with
per-observation metadata needed by the decoder and the API.

Preprocessing is fit on training trips only (feature scaling is fit
inside each model pipeline on its training fold, so no test information
leaks into preprocessing).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import candidate_miss_rate, generate_candidates
from .config import DATA_PROCESSED, DATA_RAW
from .data_processing import clean_track, discover_trips, load_trip
from .features import build_feature_rows
from .labeling import assign_reference_arcs
from .road_network import RoadNetwork

FEATURES_PARQUET = DATA_PROCESSED / "features.pkl"  # cached feature table
TRIPS_PICKLE = DATA_PROCESSED / "trips.pkl"  # networks + tracks + labels


def build_dataset(raw_dir: str | Path = DATA_RAW, force: bool = False):
    """Build (or load cached) candidate features for every discovered trip.

    Returns
    -------
    features : DataFrame, one row per (trip, observation, candidate) with
               feature columns, label column `is_match`, plus obs_idx and
               trip_id.
    store    : dict trip_id -> {track, network, ref_arc_idx, cand_lists}
               (kept in memory for decoding; also pickled for the API).
    """
    if FEATURES_PARQUET.exists() and TRIPS_PICKLE.exists() and not force:
        features = pd.read_pickle(FEATURES_PARQUET)
        with open(TRIPS_PICKLE, "rb") as f:
            store = pickle.load(f)
        return features, store

    infos = discover_trips(raw_dir)
    if not infos:
        raise FileNotFoundError(
            f"No trips (*.track) found under {raw_dir}. "
            "Place the dataset there or run scripts/make_synthetic_data.py.")

    # Incremental per-trip checkpoints: a killed run resumes instead of
    # restarting (the sandbox has limited RAM and big real networks).
    parts_dir = DATA_PROCESSED / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    if not force:
        for p in sorted(parts_dir.glob("*.pkl")):
            with open(p, "rb") as f:
                done[p.stem] = pickle.load(f)
        if done:
            print(f"  resuming: {len(done)} trip parts already built")

    all_rows = []
    store = {}
    quality = []
    for info in infos:
        if info["trip_id"] in done:
            part = done[info["trip_id"]]
            all_rows.append(part["rows"])
            store[info["trip_id"]] = part["entry"]
            quality.append(part["quality"])
            continue
        trip = load_trip(info)
        from .data_processing import validate_trip
        qrow = {"trip_id": trip.trip_id, **validate_trip(trip)}
        quality.append(qrow)

        track = clean_track(trip.track)
        if len(track) < 3:
            continue
        network = RoadNetwork(trip.nodes, trip.arcs)

        # arc geometry by arc_id, needed by the route alignment
        # (dict lookups - .loc per arc is far too slow on 100k-arc networks)
        lat_of = trip.nodes.set_index("node_id")["lat"].to_dict()
        lon_of = trip.nodes.set_index("node_id")["lon"].to_dict()
        arc_geoms = {}
        for a, u, v in zip(trip.arcs["arc_id"].to_numpy(),
                           trip.arcs["from_node"].to_numpy(),
                           trip.arcs["to_node"].to_numpy()):
            arc_geoms[int(a)] = ((lat_of[u], lon_of[u]), (lat_of[v], lon_of[v]))

        ref_arc_ids = assign_reference_arcs(track["lat"], track["lon"],
                                            arc_geoms, trip.route)
        ref_arc_idx = np.array([network.id_to_idx.get(a, -1) for a in ref_arc_ids])

        cand_lists, relaxed = generate_candidates(track["lat"], track["lon"], network)
        rows = build_feature_rows(track, network, cand_lists, ref_arc_idx)
        rows["trip_id"] = trip.trip_id
        rows["relaxed_radius"] = rows["obs_idx"].map(
            dict(zip(np.arange(len(track)), relaxed))).fillna(False)
        split = Path(info["track"]).parent.parent.name
        if split not in ("student_test", "evaluator_test") and len(rows) > 0:
            keep_obs = np.unique(rows["obs_idx"].to_numpy())
            max_keep = 240
            if len(keep_obs) > max_keep:
                stride = int(np.ceil(len(keep_obs) / max_keep))
                keep_obs = set(keep_obs[::stride].tolist())
                rows = rows[rows["obs_idx"].isin(keep_obs)]

        all_rows.append(rows)

        # Memory: training trips are never decoded, so keep only what the
        # miss-rate metric needs. Decoded trips keep the full network but
        # with adjacency as sorted lists (sets of ints cost ~5x RAM).
        network.adjacency = [sorted(s) for s in network.adjacency]
        if split in ("student_test", "evaluator_test"):
            entry = {
                "trip_id": trip.trip_id,
                "track": track,
                "network": network,
                "ref_arc_idx": ref_arc_idx,
                "ref_arc_ids": ref_arc_ids,
                "cand_lists": cand_lists,
                "route": trip.route,
            }
        else:
            entry = {
                "trip_id": trip.trip_id,
                "ref_arc_idx": ref_arc_idx,
                "cand_lists": cand_lists,
            }
        store[trip.trip_id] = entry
        with open(parts_dir / f"{trip.trip_id}.pkl", "wb") as f:
            pickle.dump({"rows": rows, "entry": entry, "quality": qrow}, f)
        del trip, track, network, arc_geoms, cand_lists, rows, lat_of, lon_of
        import gc; gc.collect()

    features = pd.concat(all_rows, ignore_index=True)
    features.to_pickle(FEATURES_PARQUET)
    with open(TRIPS_PICKLE, "wb") as f:
        pickle.dump(store, f)
    pd.DataFrame(quality).to_csv(DATA_PROCESSED / "data_quality.csv", index=False)
    return features, store


def overall_miss_rate(store) -> float:
    """Candidate Miss Rate across all trips with reference labels."""
    rates, weights = [], []
    for tid, s in store.items():
        valid = np.asarray(s["ref_arc_idx"]) >= 0
        if valid.sum() == 0:
            continue
        rates.append(candidate_miss_rate(s["cand_lists"], s["ref_arc_idx"]))
        weights.append(int(valid.sum()))
    if not rates:
        return float("nan")
    return float(np.average(rates, weights=weights))
