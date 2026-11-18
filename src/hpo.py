"""Bayesian hyperparameter optimisation of the selected Random Forest.

Strategy: Optuna's TPE (Tree-structured Parzen Estimator) sampler, a
Bayesian search that models the objective from past trials and proposes
promising configurations.

Objective: mean GroupKFold(5) log loss on the 80 training trips
(probability quality is what the HMM emission model consumes, so we
optimise calibrated error rather than raw accuracy).

The evaluator-test partition is never used anywhere in this stage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from .config import CFG, FEATURE_COLUMNS, RESULTS
from .metrics import binary_metrics, route_continuity, top1_accuracy, top1_predictions
from .sampling import subsample_for_training

SEARCH_SPACE_DOC = {
    "n_estimators": "int, 100-500: number of trees; more trees stabilise probabilities",
    "max_depth": "int, 4-32: tree depth cap; controls overfitting on noisy GPS",
    "min_samples_split": "int, 2-20: minimum samples to split a node",
    "min_samples_leaf": "int, 1-10: leaf smoothing, improves probability estimates",
    "max_features": "categorical {sqrt, log2, 1.0}: feature subsampling per split",
    "class_weight": "categorical {None, balanced}: compensate 1:4 label imbalance",
}


def _objective_factory(X, y, groups, n_folds):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features",
                                                      ["sqrt", "log2", 1.0]),
            "class_weight": trial.suggest_categorical("class_weight",
                                                      ["none", "balanced"]),
        }
        cw = None if params["class_weight"] == "none" else "balanced"
        gkf = GroupKFold(n_splits=n_folds)
        losses = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            m = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                class_weight=cw, random_state=CFG.seed, n_jobs=2)
            m.fit(X.iloc[tr_idx], y[tr_idx])
            p = m.predict_proba(X.iloc[va_idx])[:, 1]
            losses.append(binary_metrics(y[va_idx], p)["log_loss"])
        return float(np.mean(losses))
    return objective


def run_hpo(features: pd.DataFrame, partitions: dict, store,
            n_trials: int = CFG.hpo_trials, out_dir: Path = RESULTS):
    """Run TPE search, refit the best configuration, compare tuned vs
    untuned on identical partitions. Returns (best_params, summary)."""
    train_df = subsample_for_training(
        features[features["trip_id"].isin(partitions["train"])])
    stest_df = features[features["trip_id"].isin(partitions["student_test"])]
    X_tr = train_df[FEATURE_COLUMNS]
    y_tr = train_df["is_match"].to_numpy(dtype=int)
    groups = train_df["trip_id"].to_numpy()
    X_te = stest_df[FEATURE_COLUMNS]
    y_te = stest_df["is_match"].to_numpy(dtype=int)

    sampler = optuna.samplers.TPESampler(seed=CFG.seed)
    # SQLite-backed study: optimisation progress survives interruption and
    # reruns resume until n_trials completed trials exist.
    storage = f"sqlite:///{out_dir / 'hpo_study.db'}"
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="pathsnap_rf_tpe", storage=storage,
                                load_if_exists=True)
    t0 = time.perf_counter()
    objective = _objective_factory(X_tr, y_tr, groups, CFG.cv_folds)
    while len(study.trials) < n_trials:
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        print(f"      HPO trial {len(study.trials)}/{n_trials}: "
              f"best CV log loss = {study.best_value:.4f}", flush=True)
    hpo_time = time.perf_counter() - t0

    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "hpo_trials.csv", index=False)

    best = dict(study.best_params)
    best_cw = None if best["class_weight"] == "none" else "balanced"

    # ---- refit tuned model on all training trips ----
    tuned = RandomForestClassifier(
        n_estimators=best["n_estimators"], max_depth=best["max_depth"],
        min_samples_split=best["min_samples_split"],
        min_samples_leaf=best["min_samples_leaf"],
        max_features=best["max_features"], class_weight=best_cw,
        random_state=CFG.seed, n_jobs=2)
    t0 = time.perf_counter()
    tuned.fit(X_tr, y_tr)
    tuned_train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    p_te = tuned.predict_proba(X_te)[:, 1]
    tuned_infer_time = time.perf_counter() - t0
    met = binary_metrics(y_te, p_te)

    top1_accs, conts = [], []
    stest_scored = stest_df.assign(prob=p_te)
    for tid, trip_rows in stest_scored.groupby("trip_id"):
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

    summary = {
        "sampler": "Optuna TPESampler (Bayesian)",
        "objective": "mean GroupKFold(5) log loss on 80 training trips (minimise)",
        "n_trials": n_trials,
        "search_space": SEARCH_SPACE_DOC,
        "best_params": best,
        "best_cv_log_loss": float(study.best_value),
        "hpo_wall_time_s": hpo_time,
        "tuned": {
            "student_test_macro_f1": met["macro_f1"],
            "student_test_f1": met["f1"],
            "student_test_precision": met["precision"],
            "student_test_recall": met["recall"],
            "student_test_log_loss": met["log_loss"],
            "student_test_top1_accuracy": float(np.mean(top1_accs)),
            "student_test_route_continuity": float(np.mean(conts)),
            "train_time_s": tuned_train_time,
            "inference_time_s": tuned_infer_time,
        },
    }
    with open(out_dir / "hpo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    joblib.dump(tuned, out_dir / "random_forest_tuned.joblib", compress=("xz", 3))
    return best, summary, tuned
