"""Stage 1 experiment: train and compare five models from five families.

Protocol (brief: experimental design):
* cross-validation is GroupKFold(n_splits=5) over TRAIN trips only, so no
  trip straddles folds;
* student-test trips are used for development-stage comparison;
* evaluator-test trips are never touched here;
* train/inference times, configurations and seeds are recorded;
* model selection uses generalisation evidence (CV Macro F1, student-test
  Top-1 accuracy and log loss), never training accuracy alone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .config import CFG, FEATURE_COLUMNS, RESULTS
from .metrics import binary_metrics, route_continuity, top1_accuracy, top1_predictions
from .models import get_model_zoo
from .sampling import subsample_for_training


def _xy(df: pd.DataFrame):
    return df[FEATURE_COLUMNS], df["is_match"].to_numpy(dtype=int)


def compare_models(features: pd.DataFrame, partitions: dict, store,
                   seed: int = CFG.seed, out_dir: Path = RESULTS) -> pd.DataFrame:
    zoo = get_model_zoo(seed)
    # Fitting uses evenly subsampled observations per trip (see
    # src/sampling.py); evaluation always uses the full student-test table.
    train_df = subsample_for_training(
        features[features["trip_id"].isin(partitions["train"])])
    stest_df = features[features["trip_id"].isin(partitions["student_test"])]
    X_tr, y_tr = _xy(train_df)
    groups = train_df["trip_id"].to_numpy()
    X_te, y_te = _xy(stest_df)

    gkf = GroupKFold(n_splits=CFG.cv_folds)
    records = []
    fitted = {}
    for name, spec in zoo.items():
        # ---- group-aware cross-validation on the 80 training trips ----
        cv_f1, cv_logloss = [], []
        for tr_idx, va_idx in gkf.split(X_tr, y_tr, groups):
            m = clone(spec["estimator"])
            m.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
            p = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
            met = binary_metrics(y_tr[va_idx], p)
            cv_f1.append(met["macro_f1"])
            cv_logloss.append(met["log_loss"])

        # ---- fit on all training trips, evaluate on student-test ----
        model = clone(spec["estimator"])
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        p_te = model.predict_proba(X_te)[:, 1]
        infer_time = time.perf_counter() - t0

        met = binary_metrics(y_te, p_te)

        # observation-level Top-1 accuracy and route continuity, per trip
        top1_accs, conts = [], []
        stest_df = stest_df.assign(prob=p_te)
        for tid, trip_rows in stest_df.groupby("trip_id"):
            top1 = top1_predictions(trip_rows, trip_rows["prob"].to_numpy())
            ref = store[tid]["ref_arc_idx"]
            acc = top1_accuracy(top1, ref)
            if not np.isnan(acc):
                top1_accs.append(acc)
            pred_seq = top1.set_index("obs_idx")["arc_idx"].reindex(
                range(len(ref))).fillna(-1).to_numpy()
            cont = route_continuity(pred_seq, store[tid]["network"])
            if not np.isnan(cont):
                conts.append(cont)

        records.append({
            "model": name,
            "family": spec["family"],
            "config": json.dumps(spec["config"]),
            "cv_macro_f1": float(np.mean(cv_f1)),
            "cv_macro_f1_std": float(np.std(cv_f1)),
            "cv_log_loss": float(np.mean(cv_logloss)),
            "student_test_macro_f1": met["macro_f1"],
            "student_test_f1": met["f1"],
            "student_test_precision": met["precision"],
            "student_test_recall": met["recall"],
            "student_test_accuracy": met["accuracy"],
            "student_test_log_loss": met["log_loss"],
            "student_test_top1_accuracy": float(np.mean(top1_accs)),
            "student_test_route_continuity": float(np.mean(conts)),
            "train_time_s": train_time,
            "inference_time_s": infer_time,
            "inference_rows_per_s": len(X_te) / max(infer_time, 1e-9),
        })
        fitted[name] = model

        # epoch-wise training history only for the MLP (report rule: no
        # artificial curves for models without training histories)
        if name == "mlp":
            clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
            if hasattr(clf, "loss_curve_"):
                pd.DataFrame({"epoch": np.arange(1, len(clf.loss_curve_) + 1),
                              "train_loss": clf.loss_curve_}).to_csv(
                    out_dir / "mlp_training_curve.csv", index=False)

    results = pd.DataFrame(records)
    results.to_csv(out_dir / "model_comparison.csv", index=False)
    with open(out_dir / "model_comparison.json", "w") as f:
        json.dump(records, f, indent=2)

    return results
