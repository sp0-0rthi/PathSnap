"""Trip-level dataset partitions: 80 train / 10 student-test / 10 evaluator-test.

Leakage prevention (dataset brief, section 7):
* no observations from the same trip appear in two partitions - the split
  is by trip id;
* the evaluator-test partition is never used for training, feature
  selection, model selection or hyperparameter choices; it is only served
  by the API at final evaluation time;
* the split is deterministic (seeded) and persisted so every stage and
  every rerun uses identical partitions.

If the dataset ships its own partition layout (folders or a metadata
file), that layout is used verbatim instead of the seeded split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CFG, DATA_PROCESSED

PARTITION_FILE = DATA_PROCESSED / "partitions.json"


def assign_partitions(trip_ids: list[str], raw_dir=None,
                      n_train: int = CFG.n_train_trips,
                      n_student: int = CFG.n_student_test_trips,
                      n_eval: int = CFG.n_eval_trips,
                      seed: int = CFG.seed) -> dict:
    """Return {"train": [...], "student_test": [...], "evaluator_test": [...]}.

    Detection order:
    1. an existing partitions.json (previously persisted);
    2. dataset-provided layout: folders named train/student_test/
       evaluator_test (or similar) under raw_dir, or a partitions.csv;
    3. a deterministic seeded 80/10/10 shuffle (fallback; expected counts
       from the brief).
    """
    if PARTITION_FILE.exists():
        with open(PARTITION_FILE) as f:
            parts = json.load(f)
        if all(t in sum(parts.values(), []) for t in trip_ids):
            return parts

    parts = None
    if raw_dir is not None:
        parts = _detect_dataset_partitions(Path(raw_dir), trip_ids)

    if parts is None:
        rng = np.random.default_rng(seed)
        ids = np.array(sorted(trip_ids))
        rng.shuffle(ids)
        n = len(ids)
        if n >= n_train + n_student + n_eval:
            n_tr, n_st, n_ev = n_train, n_student, n_eval
        else:
            # proportional 80/10/10 fallback for small stand-in datasets
            n_tr = max(1, int(round(0.8 * n)))
            n_st = max(1, int(round(0.1 * n)))
            n_ev = max(1, n - n_tr - n_st)
            while n_tr + n_st + n_ev > n:
                n_tr -= 1
        parts = {
            "train": sorted(ids[:n_tr].tolist()),
            "student_test": sorted(ids[n_tr:n_tr + n_st].tolist()),
            "evaluator_test": sorted(ids[n_tr + n_st:n_tr + n_st + n_ev].tolist()),
        }

    PARTITION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PARTITION_FILE, "w") as f:
        json.dump(parts, f, indent=2)
    return parts


def _detect_dataset_partitions(raw_dir: Path, trip_ids: list[str]):
    """Use the dataset's own partition structure when present."""
    name_map = {
        "train": "train", "training": "train",
        "student_test": "student_test", "student_testing": "student_test",
        "student-test": "student_test", "test": "student_test",
        "evaluator_test": "evaluator_test", "evaluator_testing": "evaluator_test",
        "evaluator-test": "evaluator_test", "eval": "evaluator_test",
        "evaluation": "evaluator_test",
    }
    parts: dict[str, list[str]] = {"train": [], "student_test": [], "evaluator_test": []}
    found = False
    for child in raw_dir.iterdir():
        if child.is_dir() and child.name.lower() in name_map:
            part = name_map[child.name.lower()]
            for trip_dir in child.iterdir():
                if trip_dir.is_dir() and trip_dir.name in trip_ids:
                    parts[part].append(trip_dir.name)
                    found = True
    if found:
        # Trips not covered by the dataset's own folders are split
        # deterministically (seeded order): first fill the training quota,
        # then the student-test quota; the evaluator partition always
        # comes only from the dataset layout.
        covered = set(sum(parts.values(), []))
        rest = sorted(t for t in trip_ids if t not in covered)
        if rest:
            rng = np.random.default_rng(42)
            rest = list(np.array(rest)[rng.permutation(len(rest))])
            need_tr = min(max(0, n_train - len(parts["train"])), len(rest))
            parts["train"].extend(rest[:need_tr])
            parts["student_test"].extend(rest[need_tr:])
        return {k: sorted(v) for k, v in parts.items()}

    # metadata file variant: partitions.csv with trip_id,partition columns
    for meta in raw_dir.glob("*partition*"):
        if meta.suffix.lower() == ".csv":
            import pandas as pd
            df = pd.read_csv(meta)
            cols = {c.lower(): c for c in df.columns}
            tid_col = cols.get("trip_id") or cols.get("trip")
            part_col = cols.get("partition") or cols.get("split")
            if tid_col and part_col:
                for _, r in df.iterrows():
                    p = str(r[part_col]).lower()
                    key = name_map.get(p, "train")
                    parts[key].append(str(r[tid_col]))
                return {k: sorted(v) for k, v in parts.items()}
    return None
