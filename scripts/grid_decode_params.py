"""Joint decode-parameter grid (stay_bonus x transition_beta [x temperature])
on a chosen partition, decode-only, no retraining.

Temperature: p' = sigmoid(logit(p)/T), applied to emissions before decoding.
Tune temperature on TRAIN trips only; stay/beta on student_test; a single
final read on evaluator_test happens in full_blend_eval.py.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, DATA_PROCESSED, FEATURE_COLUMNS  # noqa: E402
from src.features import build_feature_rows  # noqa: E402
from src.hmm import trip_emissions, viterbi_decode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def temper(emissions, T):
    if T == 1.0:
        return emissions
    out = []
    for e in emissions:
        if not len(e):
            out.append(e)
            continue
        p = np.clip(e, 1e-9, 1 - 1e-9)
        logit = np.log(p / (1 - p)) / T
        out.append(1.0 / (1.0 + np.exp(-logit)))
    return out


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "student_test"
    stays = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1.5]
    betas = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [15.0]
    temps = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1.0]
    n_max = int(sys.argv[5]) if len(sys.argv) > 5 else 10

    parts = json.load(open(DATA_PROCESSED / "partitions.json"))
    trip_ids = parts[split][:n_max]
    model = joblib.load(ROOT / "artifacts" / "random_forest_tuned.joblib")

    trips = []
    for tid in trip_ids:
        with open(DATA_PROCESSED / "parts" / f"{tid}.pkl", "rb") as f:
            s = pickle.load(f)["entry"]
        rows = build_feature_rows(s["track"], s["network"], s["cand_lists"],
                                  s["ref_arc_idx"])
        em = trip_emissions(model, rows, s["cand_lists"], FEATURE_COLUMNS)
        trips.append((tid, s, em))
    print(f"prepared {len(trips)} {split} trips", flush=True)

    best = None
    for T in temps:
        for stay in stays:
            for beta in betas:
                cfg = replace(CFG, stay_bonus=stay, transition_beta=beta)
                t0 = time.perf_counter()
                hits = tot = 0
                for tid, s, em in trips:
                    pred = viterbi_decode(temper(em, T), s["cand_lists"],
                                          s["track"], s["network"], cfg)
                    ref = np.asarray(s["ref_arc_idx"])
                    v = ref >= 0
                    hits += int((pred[v] == ref[v]).sum())
                    tot += int(v.sum())
                acc = hits / tot
                tag = f"T={T} stay={stay} beta={beta}"
                print(f"{tag:<28} pooled_top1={acc:.4f} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
                if best is None or acc > best[0]:
                    best = (acc, T, stay, beta)
    print(f"BEST: top1={best[0]:.4f} T={best[1]} stay={best[2]} "
          f"beta={best[3]}", flush=True)


if __name__ == "__main__":
    main()
