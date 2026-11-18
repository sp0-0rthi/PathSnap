"""Shared evaluation metrics.

Candidate-level metrics treat each (observation, candidate) row as a
binary problem (is this candidate the reference arc?). Observation-level
Top-1 accuracy compares the highest-scoring candidate per observation
with the reference arc. Route continuity measures how often consecutive
predictions are connected in the road network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score)


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        out["log_loss"] = float(log_loss(y_true, np.clip(y_prob, 1e-9, 1 - 1e-9),
                                         labels=[0, 1]))
    except ValueError:
        out["log_loss"] = float("nan")
    return out


def top1_predictions(features: pd.DataFrame, probs) -> pd.DataFrame:
    """Pick the highest-probability candidate per observation.

    Returns a DataFrame with one row per observation: obs_idx, chosen
    arc_idx and its probability.
    """
    df = features[["obs_idx", "arc_idx"]].copy()
    df["prob"] = probs
    idx = df.groupby("obs_idx")["prob"].idxmax()
    return df.loc[idx].sort_values("obs_idx").reset_index(drop=True)


def top1_accuracy(top1: pd.DataFrame, ref_arc_idx) -> float:
    ref = np.asarray(ref_arc_idx)
    chosen = top1.set_index("obs_idx")["arc_idx"]
    valid = [i for i in chosen.index if i < len(ref) and ref[i] >= 0]
    if not valid:
        return float("nan")
    hits = sum(1 for i in valid if chosen.loc[i] == ref[i])
    return hits / len(valid)


def route_continuity(pred_arc_idx, network) -> float:
    """Fraction of consecutive predicted arcs that are connected.

    A pair counts as continuous when both arcs are identical, adjacent in
    the connectivity graph, or joined by a network path no longer than
    three times the straight GPS gap plus a buffer. This is the
    sequence-level structural quality measure the HMM is meant to improve.
    """
    pred = np.asarray(pred_arc_idx)
    if len(pred) < 2:
        return float("nan")
    ok, total = 0, 0
    for a, b in zip(pred[:-1], pred[1:]):
        if a < 0 or b < 0:
            continue
        a, b = int(a), int(b)  # reindex/fillna can yield float64
        total += 1
        if a == b or b in network.adjacency[a]:
            ok += 1
        else:
            cutoff = 3.0 * (network.length_m[a] + network.length_m[b]) + 200.0
            if network.path_distance(a, b, cutoff) is not None:
                ok += 1
    return ok / total if total else float("nan")
