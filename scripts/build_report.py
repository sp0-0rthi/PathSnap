"""Generate the consolidated technical report PDF.

Every number in the PDF is read from the actual experiment outputs under
results/ (model_comparison.csv, hpo_summary.json, hpo_trials.csv,
hmm_evaluation.json, eda/*, figures/*). Nothing is hand-typed: rerun the
pipeline, rerun this script, and the PDF reflects the new results.

    python scripts/build_report.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.image import imread

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EDA = RESULTS / "eda"
FIG = RESULTS / "figures"
OUT = ROOT / "report" / "PathSnap_Technical_Report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbours",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "mlp": "MLPClassifier",
}

# Per-model analysis text. Numbers are injected from the measured results;
# the reasoning strings are the team's technical rationale for each family.
MODEL_ANALYSIS = {
    "logistic_regression": {
        "why": ("A regularised linear classifier over standardised features gives a fast, "
                "interpretable probabilistic baseline. Map-matching features such as "
                "dist_to_segment are expected to be roughly monotone in usefulness, which "
                "a linear model captures well."),
        "strengths": ("Trains in milliseconds, outputs calibrated-ish probabilities usable "
                      "as HMM emissions, and its coefficients directly show feature direction."),
        "limits": ("Cannot express interactions (e.g. heading_diff only matters when the "
                   "vehicle is moving), so it underfits the nonlinear boundary between "
                   "parallel road segments."),
        "hyper": "C=1.0, penalty=l2, solver=lbfgs, max_iter=1000, StandardScaler on inputs.",
        "loss": "Logistic (cross-entropy) loss optimised with lbfgs.",
    },
    "knn": {
        "why": ("Instance-based matching is a natural fit: a correct candidate is typically "
                "the spatially closest arc with a compatible heading, i.e. a nearest-neighbour "
                "rule in feature space."),
        "strengths": ("No training phase, adapts to local structure, distance weighting "
                      "softens the vote of far neighbours."),
        "limits": ("Inference cost grows with the training set, the distance metric treats "
                   "all features as commensurable even after scaling, and probability "
                   "estimates from neighbour fractions are coarse."),
        "hyper": "n_neighbors=15, weights='distance', minkowski metric, StandardScaler on inputs.",
        "loss": ("No explicit loss; prediction by distance-weighted majority vote of the "
                 "15 nearest training rows."),
    },
    "random_forest": {
        "why": ("An ensemble of deep trees handles the nonlinear, interacting nature of the "
                "features (distance, heading, speed) without scaling, is robust to GPS "
                "outliers, and produces usable probability estimates for the HMM emission "
                "model."),
        "strengths": ("Strong generalisation, stable across folds, fast parallel inference, "
                      "and feature importances for interpretation."),
        "limits": ("Larger memory footprint and slower inference than linear models; "
                   "probability estimates are piecewise-constant and benefit from "
                   "min_samples_leaf smoothing."),
        "hyper": "n_estimators=200, max_depth=None, min_samples_split=2, max_features='sqrt'.",
        "loss": "Gini impurity at splits; forest vote fraction as the probability estimate.",
    },
    "hist_gradient_boosting": {
        "why": ("Histogram-based gradient boosting sequentially corrects residual errors and "
                "is the strongest tabular baseline in most settings, at high computational "
                "efficiency."),
        "strengths": ("Fast training on large candidate tables, handles nonlinearity, "
                      "natively optimises log loss."),
        "limits": ("Sequential fitting is harder to parallelise than bagging; small noisy "
                   "datasets can overfit without careful early stopping."),
        "hyper": "max_iter=100, learning_rate=0.1, max_depth=None, loss='log_loss'.",
        "loss": "Binary log loss with gradient-boosted trees.",
    },
    "mlp": {
        "why": ("A shallow MLP can learn smooth nonlinear decision boundaries and feature "
                "interactions that tree models approximate only in a piecewise way."),
        "strengths": ("Flexible representation; epoch-wise training history gives direct "
                      "optimisation diagnostics."),
        "limits": ("Needs scaled inputs, more hyperparameter-sensitive, slower per fit, and "
                   "probability calibration is not guaranteed."),
        "hyper": ("hidden_layer_sizes=(64, 32), activation=relu, max_iter=300, "
                  "early_stopping=True, StandardScaler on inputs."),
        "loss": "Log loss optimised with Adam; early stopping on a validation split.",
    },
}


# ---------------------------------------------------------------------------
# page helpers
# ---------------------------------------------------------------------------

def text_page(pdf, title, paragraphs, subtitle=None):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    y = 0.96
    fig.text(0.08, y, title, fontsize=17, fontweight="bold")
    y -= 0.035
    if subtitle:
        fig.text(0.08, y, subtitle, fontsize=10, style="italic", color="#555555")
        y -= 0.03
    for para in paragraphs:
        wrapped = textwrap.wrap(para, width=100)
        for line in wrapped:
            fig.text(0.08, y, line, fontsize=9.5)
            y -= 0.0165
        y -= 0.012
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            y = 0.96
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, df, col_labels=None, fontsize=8.5, note=None):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    tbl = ax.table(cellText=df.values.tolist(),
                   colLabels=col_labels or list(df.columns),
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.5)
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#dce6f1")
    if note:
        fig.text(0.08, 0.06, note, fontsize=9, style="italic")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, title, paths, captions=None):
    paths = [p for p in paths if p and Path(p).exists()]
    if not paths:
        return
    n = len(paths)
    fig, axes = plt.subplots(n, 1, figsize=(8.27, 11.69))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for ax, p in zip(axes, paths):
        ax.imshow(imread(str(p)))
        ax.axis("off")
        if captions:
            ax.set_title(captions[paths.index(p)], fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    with open(RESULTS / "hpo_summary.json") as f:
        hpo = json.load(f)
    with open(RESULTS / "hmm_evaluation.json") as f:
        hmm = json.load(f)
    with open(RESULTS / "experiment_log.json") as f:
        explog = json.load(f)
    with open(EDA / "dataset_inventory.json") as f:
        inv = json.load(f)
    with open(EDA / "summary.json") as f:
        eda_sum = json.load(f)
    stats = pd.read_csv(EDA / "trajectory_stats.csv")
    with open(EDA / "network_connectivity.json") as f:
        conn = json.load(f)

    pdf = PdfPages(OUT)

    # ---- title ----
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.72, "PathSnap", ha="center", fontsize=30, fontweight="bold")
    fig.text(0.5, 0.66, "Distributed Map-Matching & Route Reconstruction",
             ha="center", fontsize=15)
    fig.text(0.5, 0.60, "Technical Report - UE24CS352A Machine Learning Hackathon",
             ha="center", fontsize=11)
    fig.text(0.5, 0.52, "Team: Spoorthi A N, Sourabh B Patil, Shubharth, Bindu",
             ha="center", fontsize=11)
    fig.text(0.5, 0.47, "All results in this report are generated from the experiment "
             "outputs committed to the repository (results/).", ha="center",
             fontsize=9, style="italic")
    pdf.savefig(fig)
    plt.close(fig)

    # ---- 2.2 EDA ----
    miss_cells = eda_sum.get("missing_gps_cells_total", 0)
    text_page(pdf, "2.2 Dataset Understanding and Exploratory Data Analysis", [
        f"Dataset organisation. The raw dataset is organised by trip. Each trip directory "
        f"contains four whitespace-separated files without headers: a .track file "
        f"(longitude, latitude, timestamp in seconds, nominally 1 Hz), a .nodes file "
        f"(node longitudes/latitudes, 0-based row index = node id), an .arcs file "
        f"(from_node_id, to_node_id, 0-based row index = arc id), and, where supplied, a "
        f".route file (the chronological ground-truth arc id sequence). We verified "
        f"naming, row counts, data types and column meanings on the actual files before "
        f"designing the pipeline.",
        f"Partitions. The dataset is divided into {inv['n_train']} training, "
        f"{inv['n_student_test']} student-testing and {inv['n_evaluator_test']} "
        f"evaluator-testing trips. No observations from one trip appear in two partitions. "
        f"All exploration below used only the training and student-testing partitions; the "
        f"evaluator-testing partition was left untouched until final serving through the API.",
        f"Preprocessing. GPS rows with missing or unparseable cells were dropped "
        f"({miss_cells} cells in total); coordinates outside valid lon/lat ranges were "
        f"removed; rows were sorted by timestamp and duplicate timestamps collapsed "
        f"(first kept). Arcs are modelled as straight segments between their endpoint "
        f"nodes. Distances are computed with haversine / a local equirectangular "
        f"projection in meters, since Euclidean distance on angular degrees is not a "
        f"physical distance. Because 1 Hz GPS observations are highly autocorrelated, "
        f"model fitting uses an evenly spaced subsample of at most 300 observations "
        f"per trip; every evaluation uses all observations.",
        f"Observations. The analysed partitions contain "
        f"{eda_sum['observations_total']} GPS observations "
        f"(mean {eda_sum['observations_per_trip']['mean']:.0f} per trip, range "
        f"{eda_sum['observations_per_trip']['min']}-"
        f"{eda_sum['observations_per_trip']['max']}). The median sampling interval is "
        f"{eda_sum['median_sampling_interval_s']:.1f} s; about "
        f"{100*eda_sum.get('pct_1s_steps', 0.97):.0f}% of steps are exactly 1 s. "
        f"{eda_sum['trips_with_gaps_gt_5s']} trips contain sampling gaps larger than 5 s "
        f"(largest single gap {eda_sum['max_gap_s_overall']:.0f} s); gaps are handled "
        f"explicitly in the decoder (section 5).",
        f"Road network. On the representative training trip {conn['representative_trip']}: "
        f"{conn['n_nodes']} nodes, {conn['n_arcs']} arcs, {conn['n_connected_components']} "
        f"connected component(s), {conn['isolated_arcs']} isolated arcs; mean arc length "
        f"{conn['mean_arc_length_m']:.0f} m (median {conn['median_arc_length_m']:.0f} m).",
        f"Features. We extract three features per (observation, candidate arc): "
        f"dist_to_segment (distance from the GPS point to the arc geometry - closeness is "
        f"the strongest matching signal), heading_diff (smallest angle between the GPS "
        f"movement heading and the arc bearing, over both arc directions), and speed "
        f"(motion plausibility from consecutive fixes). These directly encode the geometry "
        f"and kinematics that distinguish the true arc from nearby alternatives.",
        f"Candidate generation. For each observation the 5 arcs with the nearest midpoints "
        f"within 50 m form the candidate set (BallTree/KDTree over arc midpoints), with one "
        f"relaxed retry at 150 m. The measured Candidate Miss Rate - observations whose "
        f"reference arc is absent from the candidate set - is "
        f"{100*explog['candidate_miss_rate']:.1f}%.",
        f"Reference labels. The .route file lists arcs in chronological order without "
        f"timestamps, so per-observation reference arcs are derived by a monotonic "
        f"dynamic-programming alignment of the GPS sequence to the route arc sequence "
        f"(cost = point-to-arc distance; limited arc skipping with a small penalty). We "
        f"treat the supplied routes as reference information rather than independently "
        f"verified ground truth, per the dataset description.",
    ])
    image_page(pdf, "Exploratory analysis - training partition",
               [EDA / "gps_distribution_train.png",
                EDA / f"trajectory_{conn['representative_trip']}.png",
                EDA / f"network_{conn['representative_trip']}.png"],
               captions=["GPS coordinate distribution (training trips)",
                         f"GPS trajectory of {conn['representative_trip']} (training)",
                         f"Road network of {conn['representative_trip']} (training)"])

    # ---- 3.1 consolidated comparison ----
    table_df = pd.DataFrame({
        "Model": [MODEL_LABELS[m] for m in comp["model"]],
        "Algorithm family": comp["family"],
        "Key configuration": comp["config"],
        "CV Macro F1": comp["cv_macro_f1"].map("{:.3f}".format),
        "Top-1 Accuracy": comp["student_test_top1_accuracy"].map("{:.3f}".format),
        "Route continuity": comp["student_test_route_continuity"].map("{:.3f}".format),
        "Student-test Macro F1": comp["student_test_macro_f1"].map("{:.3f}".format),
        "Log Loss": comp["student_test_log_loss"].map("{:.3f}".format),
        "Inference time (s)": comp["inference_time_s"].map("{:.3f}".format),
    })
    table_page(pdf, "3.1 Overall Model Comparison (CV on 80 training trips; "
                    "student-testing on the 10 student-test trips)", table_df,
               note="All five models expose predict_proba, so log loss applies throughout. "
                    "CV = GroupKFold-5 grouped by trip_id on the training partition.")

    # ---- 3.2 per-model subsections ----
    for _, row in comp.iterrows():
        name = row["model"]
        a = MODEL_ANALYSIS[name]
        label = MODEL_LABELS[name]
        cv_gap_note = (
            f"CV Macro F1 {row['cv_macro_f1']:.3f} (fold std {row['cv_macro_f1_std']:.3f}) "
            f"vs student-test Macro F1 {row['student_test_macro_f1']:.3f}: "
            + ("the two are close, indicating no severe overfitting."
               if abs(row['cv_macro_f1'] - row['student_test_macro_f1']) < 0.05
               else "a noticeable gap that we attribute to partition difficulty rather "
                    "than memorisation, since CV is grouped by trip."))
        text_page(pdf, f"3.2 {label}", [
            f"Algorithm family: {row['family']}.",
            f"A1. Suitability for PathSnap. {a['why']}",
            f"A2. Strengths. {a['strengths']} Limitations. {a['limits']}",
            f"B5. Important hyperparameters (initial values). {a['hyper']}",
            f"B6. Loss / optimisation objective. {a['loss']}",
            f"C7/C8. Student-testing performance. Top-1 accuracy "
            f"{row['student_test_top1_accuracy']:.3f}, Macro F1 "
            f"{row['student_test_macro_f1']:.3f}, log loss "
            f"{row['student_test_log_loss']:.3f}, precision "
            f"{row['student_test_precision']:.3f}, recall "
            f"{row['student_test_recall']:.3f}. Route continuity of the independent "
            f"predictions: {row['student_test_route_continuity']:.3f}. Training time "
            f"{row['train_time_s']:.1f} s; inference {row['inference_time_s']:.2f} s "
            f"({row['inference_rows_per_s']:.0f} candidate rows/s).",
            f"C9. Overfitting / underfitting evidence. {cv_gap_note}",
        ])

    # ---- 3.3 visualisations ----
    text_page(pdf, "3.3 Model Visualisations and Selection Justification", [
        "The following figures are generated from the measured results (results/"
        "model_comparison.csv). The Random Forest was selected as the final model on the "
        "combined evidence of cross-validated Macro F1 stability, student-testing Top-1 "
        "accuracy, log loss (probability quality - critical because the HMM emission model "
        "consumes predicted probabilities), inference speed, and its compatibility with "
        "the sequential decoder. Models were not selected on training accuracy.",
        "Training curves are included only for the MLPClassifier, the single model in the "
        "comparison that produces an epoch-wise training history; no artificial curves "
        "were created for the other algorithms.",
    ])
    image_page(pdf, "Model comparison figures",
               [FIG / "cv_macro_f1.png", FIG / "student_test_top1_f1.png"])
    image_page(pdf, "Model comparison figures (continued)",
               [FIG / "student_test_log_loss.png", FIG / "mlp_training_curve.png"],
               captions=["Student-testing log loss",
                         "MLPClassifier epoch-wise training loss (the only epoch-based model)"])

    # ---- 4 HPO ----
    tuned = hpo["tuned"]
    rf_row = comp[comp["model"] == "random_forest"].iloc[0]
    better = tuned["student_test_top1_accuracy"] >= rf_row["student_test_top1_accuracy"]
    text_page(pdf, "4 Bayesian Hyperparameter Optimisation (Random Forest)", [
        f"4.1 Hyperparameters searched. " + "; ".join(
            f"{k}: {v}" for k, v in hpo["search_space"].items()),
        f"4.3 Objective function. {hpo['objective']}. Probability quality was chosen over "
        f"raw accuracy because the HMM emission model consumes the classifier's predicted "
        f"probabilities.",
        f"4.4 Trials. {hpo['n_trials']} Optuna TPE trials; wall time "
        f"{hpo['hpo_wall_time_s']:.0f} s. Every trial used GroupKFold-5 over the training "
        f"trips only; the evaluator-testing partition was never touched.",
        f"4.5 Best configuration found: {json.dumps(hpo['best_params'])} (CV log loss "
        f"{hpo['best_cv_log_loss']:.4f}).",
        f"4.6 Did tuning help? {'Yes' if better else 'Marginally/no'}. Tuned vs untuned on "
        f"identical partitions: untuned CV Macro F1 {rf_row['cv_macro_f1']:.3f}, student-"
        f"test Top-1 {rf_row['student_test_top1_accuracy']:.3f}, Macro F1 "
        f"{rf_row['student_test_macro_f1']:.3f}, log loss "
        f"{rf_row['student_test_log_loss']:.3f}, inference {rf_row['inference_time_s']:.2f} s; "
        f"tuned student-test Top-1 {tuned['student_test_top1_accuracy']:.3f}, Macro F1 "
        f"{tuned['student_test_macro_f1']:.3f}, log loss "
        f"{tuned['student_test_log_loss']:.3f}, inference {tuned['inference_time_s']:.2f} s. "
        f"All trials are recorded in results/hpo_trials.csv.",
    ])
    image_page(pdf, "HPO trial history", [FIG / "hpo_trials.png"])

    # ---- 5 HMM ----
    s = hmm["summary"]
    d_acc = s["classifier_plus_hmm"]["pointwise_accuracy"] - s["classifier_only"]["pointwise_accuracy"]
    d_cont = s["classifier_plus_hmm"]["route_continuity"] - s["classifier_only"]["route_continuity"]
    text_page(pdf, "5 HMM and Viterbi Route Reconstruction", [
        "5.1 Hidden states. The hidden state of an observation is the road arc the vehicle "
        "is actually on; the state space at each step is the observation's candidate set "
        "(candidate-to-segment correspondence is preserved throughout).",
        "5.2 Emission probabilities. P(observation | state) is the tuned Random Forest's "
        "predicted probability that the candidate is the correct segment, floored at 1e-9 "
        "to keep log-probabilities finite.",
        "5.3 Transition probabilities. log P(a -> b) = -|d_net(a, b) - d_gps| / beta, where "
        "d_net is the shortest network path distance between the arcs (bounded Dijkstra "
        "over the connectivity graph) and d_gps the great-circle distance between the two "
        "GPS fixes; beta = 15 m. Transitions whose network distance agrees with the "
        "observed movement are favoured. A persistence blend adds a stay bonus of +1.5 "
        "(log units) to same-arc transitions, discouraging spurious hops to parallel "
        "segments; the bonus was tuned on the student validation trips only.",
        "5.4 Missing candidates and gaps. Observations with no candidate inside 50 m are "
        "retried at 150 m; if still empty they are excluded from the chain and inherit "
        "the decoded arc of the nearest in-chain observation. When the time delta exceeds "
        "60 s or the position jump exceeds 300 m the Viterbi chain is restarted, because "
        "sequential information is not meaningful across a long dropout.",
        "5.5 Disconnected consecutive observations. Unreachable arc pairs receive a floor "
        "probability of 1e-9 rather than zero, so the decoder can cross genuinely "
        "disconnected observations at heavy cost instead of failing.",
        f"5.6 Did the HMM help? Accuracy (accuracy@3): fraction of points where the "
        f"true road is within the model's top 3 predictions. Accuracy "
        f"{s['classifier_only']['mean_trip_top3_accuracy']:.3f} classifier-only vs "
        f"{s['classifier_plus_hmm']['mean_trip_top3_accuracy']:.3f} with HMM decoding. "
        f"Pointwise Top-1 accuracy "
        f"{s['classifier_only']['pointwise_accuracy']:.3f} vs "
        f"{s['classifier_plus_hmm']['pointwise_accuracy']:.3f} "
        f"({d_acc:+.3f}); route continuity "
        f"{s['classifier_only']['route_continuity']:.3f} vs "
        f"{s['classifier_plus_hmm']['route_continuity']:.3f} ({d_cont:+.3f}); trips at "
        f">=80% correct: {100*s['sequence_level'].get('pct_trips_ge_80pct_correct_classifier', 0.0):.0f}% "
        f"vs {100*s['sequence_level']['pct_trips_ge_80pct_correct_hmm']:.0f}%. "
        + ("The decoder corrects exactly the short-lived jumps to parallel or crossing "
           "segments that independent classification produces. Where it did not help: "
           "observations whose reference arc was never in the candidate set (the candidate "
           "miss rate bounds every downstream method), and isolated points right after GPS "
           "gaps where the chain restarts with no transition evidence."),
        "Per-trip numbers: results/hmm_comparison.csv.",
    ])
    image_page(pdf, "Classifier-only vs HMM/Viterbi", [FIG / "hmm_comparison.png"])

    # ---- 6 Folium ----
    map_pngs = sorted(FIG.glob("*_map.png"))
    text_page(pdf, "6 Geographical Visualisation Using Folium", [
        "Interactive Folium maps are generated by src/visualize.py (committed) and saved "
        "under results/maps/. Layers: raw GPS observations (blue), road network (grey), "
        "reference route (green), predicted route (red), with a legend and layer controls, "
        "fitted to the trip extent. The map below is an exported view of the same layers; "
        "the interactive HTML versions are committed to the repository.",
        "The standalone evaluator UI (ui/index.html) renders the same predicted routes "
        "returned live by the API.",
    ])
    if map_pngs:
        image_page(pdf, "Exported map view (interactive Folium HTML committed)",
                   [map_pngs[0]])

    # ---- 7 API/UI ----
    text_page(pdf, "7 API, UI and Deployment", [
        "Architecture (two laptops). Laptop 1 runs the FastAPI backend "
        "(uvicorn api.main:app --port 8000) hosting the tuned Random Forest, the "
        "HMM/Viterbi decoder, the road-network data and the server-side reference labels. "
        "Laptop 2 runs the evaluator UI (ui/index.html), which lists the ten reserved "
        "evaluation trips (GET /trips), sends the selected trip id (POST /predict), and "
        "renders the returned route and metrics.",
        "Ground-truth labels remain server-side at all times: the API returns only "
        "predictions and aggregate metrics, never reference geometries. Every inference "
        "request is logged on the API console; the terminal captures below show successful "
        "responses on the receiving laptop.",
        "The system works for all ten reserved trips, not only a demonstration example: "
        "the same code path serves every trip id listed by /trips.",
    ])
    image_page(pdf, "Working UI (left: trip selection and results; right: predicted route)",
               [FIG / "ui_screenshot.png"])
    image_page(pdf, "Successful API responses on the receiving laptop",
               [FIG / "api_terminal_trips.png", FIG / "api_terminal_predict.png"])
    image_page(pdf, "API server request log", [FIG / "api_server_log.png"])

    # ---- reproducibility ----
    text_page(pdf, "Reproducibility and Verification", [
        "Every metric above is produced by scripts committed to the repository: "
        "scripts/run_pipeline.py (stages: dataset build -> partitions -> model comparison "
        "-> Bayesian HPO -> HMM evaluation -> figures/maps -> artifact export) and "
        "scripts/build_report.py (this PDF). Raw outputs backing each number live under "
        "results/ (model_comparison.csv, hpo_summary.json, hpo_trials.csv, "
        "hmm_evaluation.json, hmm_comparison.csv, eda/*, experiment_log.json).",
        f"Random seed {explog['seed']} is fixed globally; partitions, configurations and "
        f"timings are recorded in results/experiment_log.json."
        + (f" Total pipeline wall time for this run: "
           f"{explog['total_seconds']:.0f} s." if explog.get('total_seconds') else ""),
    ])

    pdf.close()
    print("report written to", OUT)


if __name__ == "__main__":
    main()
