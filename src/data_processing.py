"""Raw dataset loading and inspection.

The raw dataset is organised BY TRIP (dataset description, section 3).
Each trip directory contains whitespace-separated files without headers:

    *.track   longitude latitude timestamp_seconds   (~1 Hz GPS, ~97% of
                                                    steps exactly 1 s)
    *.nodes   longitude latitude                    (0-based row = node id)
    *.arcs    from_node_id to_node_id               (0-based row = arc id)
    *.route   arc_id per line, chronological        (reference route,
                                                    where supplied)

A row of the GPS file does NOT correspond to a row of a network file; the
relationships are derived (see labeling.py for the observation -> arc
assignment along the reference route).

The loader auto-detects this layout anywhere under data/raw, tolerates
missing .route files, and validates types/ranges so malformed rows are
reported instead of silently propagated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_RAW


@dataclass
class Trip:
    """One trip: raw GPS observations + its road network + reference route."""

    trip_id: str
    track: pd.DataFrame          # columns: lon, lat, timestamp
    nodes: pd.DataFrame          # columns: node_id, lon, lat
    arcs: pd.DataFrame           # columns: arc_id, from_node, to_node
    route: list[int] = field(default_factory=list)  # chronological arc ids
    source_dir: str = ""

    @property
    def has_route(self) -> bool:
        return len(self.route) > 0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_trips(raw_dir: str | Path = DATA_RAW) -> list[dict]:
    """Find every trip under raw_dir.

    A trip is any directory containing exactly one *.track file; the
    sibling *.nodes / *.arcs / *.route files (any stem) complete it.
    Returns a sorted list of dicts with file paths.
    """
    raw_dir = Path(raw_dir)
    trips = []
    for track_path in sorted(raw_dir.rglob("*.track")):
        d = track_path.parent
        nodes = sorted(d.glob("*.nodes"))
        arcs = sorted(d.glob("*.arcs"))
        route = sorted(d.glob("*.route"))
        trips.append({
            "trip_id": d.name,
            "track": track_path,
            "nodes": nodes[0] if nodes else None,
            "arcs": arcs[0] if arcs else None,
            "route": route[0] if route else None,
        })
    return trips


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def load_track(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    return df


def load_nodes(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat"])
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["node_id"] = np.arange(len(df), dtype=int)  # 0-based row index = id
    return df[["node_id", "lon", "lat"]]


def load_arcs(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["from_node", "to_node"])
    df["from_node"] = pd.to_numeric(df["from_node"], errors="coerce").astype("Int64")
    df["to_node"] = pd.to_numeric(df["to_node"], errors="coerce").astype("Int64")
    df["arc_id"] = np.arange(len(df), dtype=int)  # 0-based row index = arc id
    return df[["arc_id", "from_node", "to_node"]]


def load_route(path) -> list[int]:
    if path is None:
        return []
    vals = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0])[0]
    return pd.to_numeric(vals, errors="coerce").dropna().astype(int).tolist()


def load_trip(info: dict) -> Trip:
    """Load one discovered trip and validate its contents."""
    track = load_track(info["track"])
    nodes = load_nodes(info["nodes"])
    arcs = load_arcs(info["arcs"])
    route = load_route(info["route"])
    trip = Trip(trip_id=info["trip_id"], track=track, nodes=nodes,
                arcs=arcs, route=route, source_dir=str(info["track"].parent))
    validate_trip(trip)
    return trip


# ---------------------------------------------------------------------------
# Validation / cleaning
# ---------------------------------------------------------------------------

def validate_trip(trip: Trip) -> dict:
    """Check coordinate plausibility, types and index integrity.

    Returns a dict of data-quality findings (also used by the EDA stage).
    Invalid GPS rows are dropped in clean_track(), never silently kept.
    """
    issues = {}
    t = trip.track
    issues["n_gps_rows"] = int(len(t))
    issues["n_missing_gps_cells"] = int(t.isna().sum().sum())
    issues["n_bad_lat"] = int(((t["lat"].abs() > 90)).sum())
    issues["n_bad_lon"] = int(((t["lon"].abs() > 180)).sum())
    ts = t["timestamp"].to_numpy(dtype=float)
    dt = np.diff(ts)
    issues["n_non_monotonic_ts"] = int((dt < 0).sum())
    issues["n_duplicate_ts"] = int((dt == 0).sum())
    issues["n_gaps_gt_5s"] = int((dt > 5).sum())
    issues["max_gap_s"] = float(dt.max()) if len(dt) else 0.0
    issues["n_nodes"] = int(len(trip.nodes))
    issues["n_arcs"] = int(len(trip.arcs))
    n_nodes = len(trip.nodes)
    bad_arc_idx = ((trip.arcs["from_node"] >= n_nodes) | (trip.arcs["to_node"] >= n_nodes)
                   | trip.arcs["from_node"].isna() | trip.arcs["to_node"].isna())
    issues["n_bad_arc_refs"] = int(bad_arc_idx.sum())
    n_arcs = len(trip.arcs)
    issues["n_bad_route_refs"] = int(sum(1 for a in trip.route if a < 0 or a >= n_arcs))
    issues["has_route"] = bool(trip.has_route)
    return issues


def clean_track(track: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid GPS rows and sort by timestamp.

    Cleaning rules (recorded in the EDA output):
    * rows with missing/unparseable lon/lat/timestamp are removed;
    * coordinates outside valid lon/lat ranges are removed;
    * rows are sorted by timestamp and duplicate timestamps are collapsed
      (first kept) so speeds and gaps are well defined.
    """
    df = track.dropna(subset=["lon", "lat", "timestamp"]).copy()
    df = df[(df["lat"].abs() <= 90) & (df["lon"].abs() <= 180)]
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    return df.reset_index(drop=True)
