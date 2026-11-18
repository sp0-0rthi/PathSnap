"""Continuation: finish pipeline stages 4-8 reusing the cached dataset and
the persisted (sqlite) Optuna study. Used to cap HPO at a given trial count
without rerunning earlier stages.

    python scripts/finish_after_hpo.py --hpo-trials 10
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
    ap.add_argument("--hpo-trials", type=int, default=10)
    args = ap.parse_args()

    from src.pipeline_data import build_dataset, overall_miss_rate
    features, store = build_dataset(DATA_RAW, force=False)
    from src.splits import assign_partitions
    partitions = assign_partitions(sorted(store.keys()), raw_dir=DATA_RAW)
    print("dataset:", features.shape, {k: len(v) for k, v in partitions.items()})

    from src.hpo import run_hpo
    best_params, hpo_summary, tuned_model = run_hpo(
        features, partitions, store, n_trials=args.hpo_trials)
    print("best params:", best_params)

    from src.evaluate import compare_decoders, decode_trip
    hmm_summary = compare_decoders(tuned_model, partitions, store)
    print(json.dumps(hmm_summary, indent=2))

    import pandas as pd
    from src.visualize import (fig_cv_macro_f1, fig_hmm_comparison,
                               fig_hpo_trials, fig_log_loss,
                               fig_mlp_training_curve, fig_top1_f1,
                               folium_trip_map, matplotlib_map)
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
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

    from src.export_artifacts import export
    meta = export(tuned_model, partitions, store, hpo_summary=hpo_summary)
    print("artifacts:", meta["n_eval_trips"], "eval trips")

    log = {"seed": CFG.seed, "config": CFG.__dict__,
           "candidate_miss_rate": overall_miss_rate(store),
           "hpo_trials": args.hpo_trials,
           "note": "continued via scripts/finish_after_hpo.py",
           "total_seconds": None}
    with open(RESULTS / "experiment_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    print("continuation done")


if __name__ == "__main__":
    main()
