"""Route-reconstruction evaluation: classifier-only vs classifier + HMM.

For every trip in the requested partition we decode twice - once with the
independent per-observation argmax (classifier-only) and once with the
Viterbi decoder over the classifier's emission probabilities - and compare
pointwise accuracy, route continuity and sequence-level behaviour against
the reference route.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG, FEATURE_COLUMNS, RESULTS
from .features import build_feature_rows
from .hmm import trip_emissions, viterbi_decode
from .metrics import route_continuity


def _accuracy(pred, ref) -> float:
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    valid = ref >= 0
    if valid.sum() == 0:
        return float("nan")
    return float((pred[valid] == ref[valid]).mean())


def decode_trip(model, trip_store, cfg: CFG = CFG):
    """Run both decoders on one trip. Returns per-observation predictions
    and the trip's candidate feature rows."""
    track = trip_store["track"]
    network = trip_store["network"]
    cand_lists = trip_store["cand_lists"]
    rows = build_feature_rows(track, network, cand_lists, trip_store["ref_arc_idx"])
    emissions = trip_emissions(model, rows, cand_lists, FEATURE_COLUMNS)

    n = len(track)
    # classifier-only: argmax emission per observation, -1 when no candidate
    clf_pred = np.full(n, -1, dtype=int)
    for i, (cands, emit) in enumerate(zip(cand_lists, emissions)):
        if len(cands):
            clf_pred[i] = cands[int(np.argmax(emit))]
    hmm_pred = viterbi_decode(emissions, cand_lists, track, network, cfg)
    return clf_pred, hmm_pred, rows


def compare_decoders(model, partitions: dict, store, split: str = "student_test",
                     out_dir: Path = RESULTS) -> dict:
    trip_ids = partitions[split]
    per_trip = []
    all_clf_correct, all_hmm_correct = [], []
    for tid in trip_ids:
        s = store[tid]
        clf_pred, hmm_pred, _ = decode_trip(model, s)
        print(f"      decoded {tid}", flush=True)
        ref = s["ref_arc_idx"]
        valid = ref >= 0
        clf_acc = _accuracy(clf_pred, ref)
        hmm_acc = _accuracy(hmm_pred, ref)
        clf_cont = route_continuity(clf_pred, s["network"])
        hmm_cont = route_continuity(hmm_pred, s["network"])
        per_trip.append({
            "trip_id": tid,
            "n_observations": int(valid.sum()),
            "classifier_top1_accuracy": clf_acc,
            "hmm_top1_accuracy": hmm_acc,
            "classifier_route_continuity": clf_cont,
            "hmm_route_continuity": hmm_cont,
        })
        all_clf_correct.append((clf_pred[valid] == ref[valid]))
        all_hmm_correct.append((hmm_pred[valid] == ref[valid]))

    df = pd.DataFrame(per_trip)
    df.to_csv(out_dir / "hmm_comparison.csv", index=False)

    clf_all = np.concatenate(all_clf_correct) if all_clf_correct else np.array([])
    hmm_all = np.concatenate(all_hmm_correct) if all_hmm_correct else np.array([])
    summary = {
        "split": split,
        "n_trips": len(trip_ids),
        "classifier_only": {
            "pointwise_accuracy": float(clf_all.mean()) if len(clf_all) else float("nan"),
            "mean_trip_accuracy": float(df["classifier_top1_accuracy"].mean()),
            "route_continuity": float(df["classifier_route_continuity"].mean()),
        },
        "classifier_plus_hmm": {
            "pointwise_accuracy": float(hmm_all.mean()) if len(hmm_all) else float("nan"),
            "mean_trip_accuracy": float(df["hmm_top1_accuracy"].mean()),
            "route_continuity": float(df["hmm_route_continuity"].mean()),
        },
        # sequence-level: distribution of per-trip accuracies
        "sequence_level": {
            "pct_trips_ge_80pct_correct_classifier":
                float((df["classifier_top1_accuracy"] >= 0.8).mean()),
            "pct_trips_ge_80pct_correct_hmm":
                float((df["hmm_top1_accuracy"] >= 0.8).mean()),
        },
    }
    with open(out_dir / "hmm_evaluation.json", "w") as f:
        json.dump({"summary": summary, "per_trip": per_trip}, f, indent=2)
    return summary
