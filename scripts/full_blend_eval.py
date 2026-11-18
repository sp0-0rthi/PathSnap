"""Full v3.1 evaluation: persistence-blend decoder on student_test and
evaluator_test (one final read on the evaluator partition).

Writes results/blend_evaluation.json and refreshes results/hmm_evaluation.json
(student_test, same schema as before plus Top-3 fields).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, DATA_PROCESSED, FEATURE_COLUMNS, RESULTS  # noqa: E402
from src.evaluate import _accuracy  # noqa: E402
from src.features import build_feature_rows  # noqa: E402
from src.hmm import trip_emissions, viterbi_decode  # noqa: E402
from src.metrics import route_continuity  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def eval_trip(model, tid, s):
    track, network = s["track"], s["network"]
    cand_lists = s["cand_lists"]
    ref = np.asarray(s["ref_arc_idx"])
    rows = build_feature_rows(track, network, cand_lists, s["ref_arc_idx"])
    emissions = trip_emissions(model, rows, cand_lists, FEATURE_COLUMNS)
    n = len(track)
    clf_pred = np.full(n, -1, dtype=int)
    for i, (c, e) in enumerate(zip(cand_lists, emissions)):
        if len(c):
            clf_pred[i] = c[int(np.argmax(e))]
    hmm_pred = viterbi_decode(emissions, cand_lists, track, network, CFG)

    t3 = t3h = nv = 0
    for i, (c, e) in enumerate(zip(cand_lists, emissions)):
        if ref[i] < 0 or not len(c):
            continue
        nv += 1
        order = [c[o] for o in np.argsort(e)[::-1][:3]]
        if ref[i] in order:
            t3 += 1
        hmm3 = [hmm_pred[i]] + [a for a in order if a != hmm_pred[i]][:2]
        if ref[i] in hmm3:
            t3h += 1
    valid = ref >= 0
    return {
        "trip_id": tid,
        "n_observations": int(valid.sum()),
        "classifier_top1_accuracy": _accuracy(clf_pred, ref),
        "hmm_top1_accuracy": _accuracy(hmm_pred, ref),
        "classifier_top3_accuracy": t3 / nv if nv else float("nan"),
        "hmm_top3_accuracy": t3h / nv if nv else float("nan"),
        "classifier_route_continuity": route_continuity(clf_pred, network),
        "hmm_route_continuity": route_continuity(hmm_pred, network),
    }, (clf_pred[valid] == ref[valid]), (hmm_pred[valid] == ref[valid])


def run_split(model, store, trip_ids, split):
    per_trip, clf_all, hmm_all = [], [], []
    for tid in trip_ids:
        t0 = time.perf_counter()
        row, c_ok, h_ok = eval_trip(model, tid, store[tid])
        per_trip.append(row)
        clf_all.append(c_ok)
        hmm_all.append(h_ok)
        print(f"  [{split}] {tid}: hmm_top1={row['hmm_top1_accuracy']:.4f} "
              f"hmm_top3={row['hmm_top3_accuracy']:.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(per_trip)
    clf_cat = np.concatenate(clf_all)
    hmm_cat = np.concatenate(hmm_all)
    summary = {
        "split": split,
        "stay_bonus": CFG.stay_bonus,
        "n_trips": len(trip_ids),
        "classifier_only": {
            "pointwise_accuracy": float(clf_cat.mean()),
            "mean_trip_accuracy": float(df["classifier_top1_accuracy"].mean()),
            "mean_trip_top3_accuracy": float(df["classifier_top3_accuracy"].mean()),
            "route_continuity": float(df["classifier_route_continuity"].mean()),
        },
        "classifier_plus_hmm": {
            "pointwise_accuracy": float(hmm_cat.mean()),
            "mean_trip_accuracy": float(df["hmm_top1_accuracy"].mean()),
            "mean_trip_top3_accuracy": float(df["hmm_top3_accuracy"].mean()),
            "route_continuity": float(df["hmm_route_continuity"].mean()),
        },
        "sequence_level": {
            "pct_trips_ge_80pct_correct_hmm":
                float((df["hmm_top1_accuracy"] >= 0.8).mean()),
        },
    }
    return summary, per_trip, df


def main():
    partitions = json.load(open(DATA_PROCESSED / "partitions.json"))
    model = joblib.load(ROOT / "artifacts" / "random_forest_tuned.joblib")
    print("loading store ...", flush=True)
    with open(DATA_PROCESSED / "trips.pkl", "rb") as f:
        store = pickle.load(f)
    print(f"store loaded ({len(store)} trips)", flush=True)

    out = {}
    for split in ("student_test", "evaluator_test"):
        summary, per_trip, df = run_split(model, store, partitions[split], split)
        out[split] = {"summary": summary, "per_trip": per_trip}
        if split == "student_test":
            df.to_csv(RESULTS / "hmm_comparison.csv", index=False)
            with open(RESULTS / "hmm_evaluation.json", "w") as f:
                json.dump({"summary": summary, "per_trip": per_trip}, f, indent=2)

    with open(RESULTS / "blend_evaluation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v["summary"] for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
