"""All plots and Folium maps.

Every figure is generated from actual experiment outputs (results/*.csv /
*.json), per the report guidelines. Folium maps use distinguishable
colours, a legend, layer controls and a fitted extent; the interactive
HTML plus this generation code are committed to the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import RESULTS

FIG_DIR = RESULTS / "figures"
MAP_DIR = RESULTS / "maps"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "mlp": "MLPClassifier",
}


def _labels(df):
    return [MODEL_LABELS.get(m, m) for m in df["model"]]


def fig_cv_macro_f1(comparison: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = comparison.sort_values("cv_macro_f1", ascending=False)
    ax.bar(_labels(d), d["cv_macro_f1"], yerr=d["cv_macro_f1_std"],
           color="#4C72B0", capsize=4)
    ax.set_ylabel("CV Macro F1-score")
    ax.set_title("Cross-validation Macro F1-score by model (GroupKFold-5, train trips)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    p = FIG_DIR / "cv_macro_f1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_top1_f1(comparison: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = comparison.sort_values("student_test_top1_accuracy", ascending=False)
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["student_test_top1_accuracy"], width=0.4,
           label="Top-1 Accuracy", color="#55A868")
    ax.bar(x + 0.2, d["student_test_macro_f1"], width=0.4,
           label="Macro F1-score", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(d), rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Student-testing Top-1 Accuracy and Macro F1-score by model")
    ax.legend()
    fig.tight_layout()
    p = FIG_DIR / "student_test_top1_f1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_log_loss(comparison: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = comparison.sort_values("student_test_log_loss")
    ax.bar(_labels(d), d["student_test_log_loss"], color="#8172B3")
    ax.set_ylabel("Log Loss (lower is better)")
    ax.set_title("Student-testing Log Loss by model")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    p = FIG_DIR / "student_test_log_loss.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_mlp_training_curve(curve_csv: Path) -> Path | None:
    curve_csv = Path(curve_csv)
    if not curve_csv.exists():
        return None
    df = pd.read_csv(curve_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"], color="#CCB974")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log loss)")
    ax.set_title("MLPClassifier training history (epoch-wise log loss)")
    fig.tight_layout()
    p = FIG_DIR / "mlp_training_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_hpo_trials(trials_csv: Path) -> Path:
    df = pd.read_csv(trials_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["number"], df["value"], c="#4C72B0", alpha=0.7, label="Trial objective (CV log loss)")
    best = df["value"].cummin()
    ax.plot(df["number"], best, color="#C44E52", label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean CV Log Loss")
    ax.set_title("Bayesian HPO (Optuna TPE) trial history - Random Forest")
    ax.legend()
    fig.tight_layout()
    p = FIG_DIR / "hpo_trials.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_hmm_comparison(hmm_eval_json: Path) -> Path:
    with open(hmm_eval_json) as f:
        data = json.load(f)
    s = data["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ["Classifier only", "Classifier + HMM"]
    acc = [s["classifier_only"]["pointwise_accuracy"],
           s["classifier_plus_hmm"]["pointwise_accuracy"]]
    cont = [s["classifier_only"]["route_continuity"],
            s["classifier_plus_hmm"]["route_continuity"]]
    axes[0].bar(labels, acc, color=["#937860", "#55A868"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Pointwise Top-1 accuracy")
    axes[1].bar(labels, cont, color=["#937860", "#55A868"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Route continuity")
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)
            tick.set_ha("right")
    fig.suptitle(f"Classifier-only vs HMM/Viterbi ({s['split']} trips)")
    fig.tight_layout()
    p = FIG_DIR / "hmm_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Folium maps
# ---------------------------------------------------------------------------

def folium_trip_map(trip_store, pred_arc_idx=None, title: str = "") -> Path:
    """Interactive map: raw GPS, road network, reference route, prediction.

    Layers (all toggleable): raw GPS observations (blue dots), road
    network (grey lines), reference route (green), predicted route (red).
    """
    import folium

    track = trip_store["track"]
    network = trip_store["network"]
    ref = np.asarray(trip_store["ref_arc_idx"])

    center = [float(track["lat"].mean()), float(track["lon"].mean())]
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    fg_net = folium.FeatureGroup(name="Road network (grey)", show=True)
    # render at most 20k arcs: real networks can exceed 100k arcs, which
    # would make the interactive map unwieldy
    arc_sel = np.linspace(0, network.n_arcs - 1,
                          min(network.n_arcs, 20000)).astype(int)
    for i in arc_sel:
        folium.PolyLine(network.arc_geometry_latlon(int(i)),
                        color="#999999", weight=2, opacity=0.5).add_to(fg_net)
    fg_net.add_to(m)

    fg_gps = folium.FeatureGroup(name="Raw GPS observations (blue)", show=True)
    pts = list(zip(track["lat"], track["lon"]))
    folium.PolyLine(pts, color="#1f77b4", weight=1, opacity=0.5).add_to(fg_gps)
    for la, lo in pts[::max(1, len(pts) // 400)]:
        folium.CircleMarker([la, lo], radius=2.5, color="#1f77b4",
                            fill=True, fill_opacity=0.7).add_to(fg_gps)
    fg_gps.add_to(m)

    fg_ref = folium.FeatureGroup(name="Reference route (green)", show=True)
    seen = set()
    for a in ref:
        if a >= 0 and a not in seen:
            seen.add(a)
            folium.PolyLine(network.arc_geometry_latlon(int(a)),
                            color="#2ca02c", weight=4, opacity=0.9).add_to(fg_ref)
    fg_ref.add_to(m)

    if pred_arc_idx is not None:
        fg_pred = folium.FeatureGroup(name="Predicted route (red)", show=True)
        seen = set()
        for a in np.asarray(pred_arc_idx):
            if a >= 0 and a not in seen:
                seen.add(int(a))
                folium.PolyLine(network.arc_geometry_latlon(int(a)),
                                color="#d62728", weight=4, opacity=0.9).add_to(fg_pred)
        fg_pred.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: white; padding: 10px 14px; border: 1px solid #888;
         border-radius: 6px; font-size: 13px;">
      <b>PathSnap{(' - ' + title) if title else ''}</b><br>
      <span style="color:#1f77b4">&#9679;</span> Raw GPS observations<br>
      <span style="color:#999999">&#9473;</span> Road network<br>
      <span style="color:#2ca02c">&#9473;</span> Reference route<br>
      <span style="color:#d62728">&#9473;</span> Predicted route
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([[float(track["lat"].min()), float(track["lon"].min())],
                  [float(track["lat"].max()), float(track["lon"].max())]])
    out = MAP_DIR / f"{trip_store.get('trip_id', 'map')}_map.html"
    m.save(str(out))
    return out


def matplotlib_map(trip_store, pred_arc_idx=None, title: str = "") -> Path:
    """Static exported view of the same layers (used as the map image
    embedded in the PDF report)."""
    track = trip_store["track"]
    network = trip_store["network"]
    ref = np.asarray(trip_store["ref_arc_idx"])

    fig, ax = plt.subplots(figsize=(8, 7))
    arc_sel = np.linspace(0, network.n_arcs - 1,
                          min(network.n_arcs, 20000)).astype(int)
    for i in arc_sel:
        g = network.arc_geometry_latlon(int(i))
        ax.plot([g[0][1], g[1][1]], [g[0][0], g[1][0]],
                color="#bbbbbb", lw=0.8, zorder=1)
    ax.plot(track["lon"], track["lat"], ".", color="#1f77b4", ms=3,
            label="Raw GPS observations", zorder=2)
    drawn = set()
    for a in ref:
        if a >= 0 and a not in drawn:
            drawn.add(a)
            g = network.arc_geometry_latlon(int(a))
            ax.plot([g[0][1], g[1][1]], [g[0][0], g[1][0]], color="#2ca02c",
                    lw=3, label="Reference route" if len(drawn) == 1 else None,
                    zorder=3)
    if pred_arc_idx is not None:
        drawn = set()
        for a in np.asarray(pred_arc_idx):
            if a >= 0 and a not in drawn:
                drawn.add(int(a))
                g = network.arc_geometry_latlon(int(a))
                ax.plot([g[0][1], g[1][1]], [g[0][0], g[1][0]], color="#d62728",
                        lw=2.5, ls="--",
                        label="Predicted route" if len(drawn) == 1 else None,
                        zorder=4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or "PathSnap trip map")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out = FIG_DIR / f"{trip_store.get('trip_id', 'trip')}_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
