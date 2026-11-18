"""Run the complete PathSnap pipeline end to end.

Stages:
    1. build the candidate-level dataset (load, clean, label, candidates,
       features)
    2. assign/persist the 80/10/10 trip partitions
    3. train + compare the five models (GroupKFold CV on train, metrics on
       student-test)
    4. Bayesian HPO of the Random Forest (Optuna TPE), tuned vs untuned
    5. HMM/Viterbi vs classifier-only comparison (student-test)
    6. figures + Folium maps
    7. export inference artifacts for the API
    8. write the experiment log (seeds, configs, timings, miss rate)

Usage:
    python scripts/run_pipeline.py                 # full run
    python scripts/run_pipeline.py --fast          # quick smoke test
    python scripts/run_pipeline.py --hpo-trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, DATA_RAW, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(DATA_RAW))
    ap.add_argument("--hpo-trials", type=int, default=CFG.hpo_trials)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="small smoke test: fewer trials")
    args = ap.parse_args()
    hpo_trials = 5 if args.fast else args.hpo_trials

    t_start = time.perf_counter()
    log = {"seed": CFG.seed, "config": CFG.__dict__, "stages": {}}

    if args.force_rebuild:
        # clear the persisted HPO study too, so no stale trials leak in
        stale = RESULTS / "hpo_study.db"
        if stale.exists():
            stale.unlink()

    from src.pipeline_data import build_dataset, overall_miss_rate
    print("[1/8] Building candidate-level dataset ...")
    t0 = time.perf_counter()
    features, store = build_dataset(args.raw_dir, force=args.force_rebuild)
    miss_rate = overall_miss_rate(store)
    log["stages"]["build_dataset"] = {"seconds": time.perf_counter() - t0}
    log["candidate_miss_rate"] = miss_rate
    print(f"      {features.shape[0]} candidate rows, "
          f"{features['trip_id'].nunique()} trips, "
          f"Candidate Miss Rate = {miss_rate:.4f}")

    print("[2/8] Assigning partitions ...")
    from src.splits import assign_partitions
    partitions = assign_partitions(sorted(store.keys()), raw_dir=args.raw_dir)
    print({k: len(v) for k, v in partitions.items()})
    log["partitions"] = {k: len(v) for k, v in partitions.items()}

    print("[3/8] Model comparison (5 families) ...")
    t0 = time.perf_counter()
    from src.train_models import compare_models
    comparison = compare_models(features, partitions, store)
    log["stages"]["compare_models"] = {"seconds": time.perf_counter() - t0}
    print(comparison[["model", "cv_macro_f1", "student_test_top1_accuracy",
                      "student_test_log_loss"]].to_string(index=False))

    print("[4/8] Bayesian HPO of Random Forest (Optuna TPE) ...")
    t0 = time.perf_counter()
    from src.hpo import run_hpo
    best_params, hpo_summary, tuned_model = run_hpo(
        features, partitions, store, n_trials=hpo_trials)
    log["stages"]["hpo"] = {"seconds": time.perf_counter() - t0,
                            "n_trials": hpo_trials, "best_params": best_params}
    print("      best params:", best_params)

    print("[5/8] HMM/Viterbi vs classifier-only (student-test) ...")
    t0 = time.perf_counter()
    from src.evaluate import compare_decoders
    hmm_summary = compare_decoders(tuned_model, partitions, store)
    log["stages"]["hmm_eval"] = {"seconds": time.perf_counter() - t0}
    print(json.dumps(hmm_summary, indent=2))

    print("[6/8] Figures and Folium maps ...")
    import pandas as pd
    from src.evaluate import decode_trip
    from src.visualize import (fig_cv_macro_f1, fig_hmm_comparison,
                               fig_hpo_trials, fig_log_loss,
                               fig_mlp_training_curve, fig_top1_f1,
                               folium_trip_map, matplotlib_map)
    fig_cv_macro_f1(comparison)
    fig_top1_f1(comparison)
    fig_log_loss(comparison)
    fig_mlp_training_curve(RESULTS / "mlp_training_curve.csv")
    fig_hpo_trials(RESULTS / "hpo_trials.csv")
    fig_hmm_comparison(RESULTS / "hmm_evaluation.json")

    demo_trip = partitions["student_test"][0]
    s = store[demo_trip]
    _, hmm_pred, _ = decode_trip(tuned_model, s)
    folium_trip_map(s, pred_arc_idx=hmm_pred,
                    title=f"{demo_trip} - predicted vs reference")
    matplotlib_map(s, pred_arc_idx=hmm_pred,
                   title=f"Trip {demo_trip}: GPS, network, reference, prediction")
    print(f"      demo map for trip {demo_trip}")

    print("[7/8] Exporting artifacts ...")
    from src.export_artifacts import export
    meta = export(tuned_model, partitions, store, hpo_summary=hpo_summary)
    print(f"      {meta['n_eval_trips']} evaluator trips packaged for the API")

    print("[8/8] Experiment log ...")
    log["total_seconds"] = time.perf_counter() - t_start
    with open(RESULTS / "experiment_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    print("Done. Results in", RESULTS)


if __name__ == "__main__":
    main()
