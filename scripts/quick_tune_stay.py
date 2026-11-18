"""Quick stay-bonus (persistence blend) tuning on student validation trips.

Loads a few student_test trip parts from data/processed/parts, decodes with
several stay_bonus values and prints Top-1 / continuity. Evaluator trips are
never touched here.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, DATA_PROCESSED, FEATURE_COLUMNS  # noqa: E402
from src.evaluate import _accuracy  # noqa: E402
from src.features import build_feature_rows  # noqa: E402
from src.hmm import trip_emissions, viterbi_decode  # noqa: E402
from src.metrics import route_continuity  # noqa: E402
import joblib  # noqa: E402


def main():
    n_trips = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    grid = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                               else ["0.0", "1.0", "2.0"])]
    parts = json.load(open(DATA_PROCESSED / "partitions.json"))
    trip_ids = parts["student_test"][:n_trips]
    model = joblib.load(Path(__file__).resolve().parent.parent
                        / "artifacts" / "random_forest_tuned.joblib")

    for tid in trip_ids:
        with open(DATA_PROCESSED / "parts" / f"{tid}.pkl", "rb") as f:
            s = pickle.load(f)["entry"]
        track, network = s["track"], s["network"]
        cand_lists, ref = s["cand_lists"], np.asarray(s["ref_arc_idx"])
        rows = build_feature_rows(track, network, cand_lists, s["ref_arc_idx"])
        emissions = trip_emissions(model, rows, cand_lists, FEATURE_COLUMNS)
        for w in grid:
            cfg = replace(CFG, stay_bonus=w)
            t0 = time.perf_counter()
            pred = viterbi_decode(emissions, cand_lists, track, network, cfg)
            acc = _accuracy(pred, ref)
            cont = route_continuity(pred, network)
            print(f"{tid} stay_bonus={w:<4} top1={acc:.4f} "
                  f"continuity={cont:.4f} ({time.perf_counter()-t0:.1f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
