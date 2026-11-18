"""Export the trained model and every inference asset the API needs.

Artifacts (committed to the repository, per the submission guidelines):
* artifacts/random_forest_tuned.joblib - THE trained model weights
  (dedicated file; the API loads exactly this file)
* artifacts/MODEL_WEIGHTS.md      - weights info: file name, tuned
                                    hyperparameters, achieved metrics
* artifacts/hmm_config.json       - HMM/Viterbi decoder configuration
* artifacts/eval_bundle.pkl       - SERVER-SIDE ONLY: the ten reserved
                                    evaluator-test trips with their road
                                    networks and reference labels. The API
                                    uses the labels solely for server-side
                                    accuracy calculation; they are never
                                    sent to the UI or the prediction model.
* artifacts/eval_trips.json       - trip ids + sizes for GET /trips
"""

from __future__ import annotations

import json
import pickle

import numpy as np
from pathlib import Path

from .config import ARTIFACTS, CFG, FEATURE_COLUMNS

def slim_network_for_bundle(net):
    """Drop geometry the API recomputes at load; downcast the rest."""
    net.adjacency = [np.asarray(a, dtype=np.int32) for a in net.adjacency]
    net._tree = None
    net.p0 = net.p0.astype(np.float32)
    net.p1 = net.p1.astype(np.float32)
    net.length_m = net.length_m.astype(np.float32)
    net.bearing = net.bearing.astype(np.float32)
    net.latlon0 = None
    net.latlon1 = None
    net.mid = None
    net.mid_lat = None
    net.mid_lon = None
    return net



def export(model, partitions: dict, store, hpo_summary: dict | None = None) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    import joblib
    # dedicated weights file - the API loads exactly this file
    joblib.dump(model, ARTIFACTS / "random_forest_tuned.joblib")
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS},
                ARTIFACTS / "model.joblib")

    if hpo_summary:
        t = hpo_summary["tuned"]
        (ARTIFACTS / "MODEL_WEIGHTS.md").write_text(f"""# Trained model weights

* **File:** `artifacts/random_forest_tuned.joblib` (scikit-learn
  RandomForestClassifier, loaded by the API at startup)
* **Model:** Random Forest, selected from a five-family comparison
  (Logistic Regression, KNN, Random Forest, HistGradientBoosting,
  MLPClassifier) and tuned with Bayesian HPO (Optuna TPE,
  {hpo_summary['n_trials']} trials, objective: {hpo_summary['objective']})
* **Features:** {FEATURE_COLUMNS}

## Tuned hyperparameters

```json
{json.dumps(hpo_summary['best_params'], indent=2)}
```

## Achieved metrics (student-testing partition)

* Top-1 accuracy: {t['student_test_top1_accuracy']:.4f}
* Macro F1: {t['student_test_macro_f1']:.4f}
* Precision / recall: {t['student_test_precision']:.4f} / {t['student_test_recall']:.4f}
* Log loss: {t['student_test_log_loss']:.4f}
* Route continuity (classifier-only): {t['student_test_route_continuity']:.4f}
* Best CV log loss during HPO: {hpo_summary['best_cv_log_loss']:.4f}

Full experiment outputs: `results/model_comparison.csv`,
`results/hpo_summary.json`, `results/hpo_trials.csv`,
`results/hmm_evaluation.json`.
""")

    hmm_cfg = {
        "transition_beta": CFG.transition_beta,
        "transition_eps": CFG.transition_eps,
        "emission_eps": CFG.emission_eps,
        "gap_seconds": CFG.gap_seconds,
        "gap_dist_m": CFG.gap_dist_m,
        "candidate_k": CFG.candidate_k,
        "candidate_radius_m": CFG.candidate_radius_m,
        "candidate_relaxed_radius_m": CFG.candidate_relaxed_radius_m,
    }
    with open(ARTIFACTS / "hmm_config.json", "w") as f:
        json.dump(hmm_cfg, f, indent=2)

    eval_ids = partitions["evaluator_test"]
    bundle = {}
    for tid in eval_ids:
        s = store[tid]
        net = slim_network_for_bundle(s["network"])
        bundle[tid] = {
            "track": s["track"],
            "network": net,
            "ref_arc_idx": s["ref_arc_idx"],
            "route": s["route"],
        }
    with open(ARTIFACTS / "eval_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    trips_meta = [{"trip_id": tid, "num_points": int(len(bundle[tid]["track"]))}
                  for tid in eval_ids]
    with open(ARTIFACTS / "eval_trips.json", "w") as f:
        json.dump({"trips": trips_meta}, f, indent=2)
    return {"eval_trips": trips_meta, "n_eval_trips": len(eval_ids)}
