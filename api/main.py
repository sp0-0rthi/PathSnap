"""PathSnap FastAPI backend - runs on Laptop 1 (the model/API laptop).

Endpoints
---------
GET  /trips    -> the ten reserved evaluation trips (ids + sizes)
POST /predict  -> {"trip_id": "..."}; runs the FULL pipeline server-side:
                  candidate generation -> feature engineering -> tuned
                  Random Forest emission probabilities -> HMM/Viterbi
                  decoding -> server-side accuracy against the internal
                  reference data. Ground truth never leaves this process:
                  only metrics and the predicted route are returned.

Every inference request is logged to the console (the evaluators ask to
see API logs on the receiving laptop).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidates import candidate_miss_rate, generate_candidates  # noqa: E402
from src.config import ARTIFACTS, CFG, FEATURE_COLUMNS  # noqa: E402
from src.evaluate import _accuracy  # noqa: E402
from src.features import build_feature_rows  # noqa: E402
from src.hmm import trip_emissions, viterbi_decode  # noqa: E402
from src.metrics import route_continuity  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

STATE: dict = {}



def _load_artifact(path):
    """Load a joblib/pickle artifact, transparently decompressing xz."""
    import lzma
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:6] == b"\xfd7zXZ\x00":
        blob = lzma.decompress(blob)
    import io
    try:
        return joblib.load(io.BytesIO(blob))
    except Exception:
        return pickle.loads(blob)

@app.on_event("startup")
def load_artifacts():
    with open(ARTIFACTS / "eval_trips.json") as f:
        STATE["trips"] = json.load(f)["trips"]
    # the API loads exactly the dedicated trained-weights file
    STATE["model"] = _load_artifact(ARTIFACTS / "random_forest_tuned.joblib")
    STATE["feature_columns"] = FEATURE_COLUMNS
    STATE["eval"] = _load_artifact(ARTIFACTS / "eval_bundle.pkl")
    for _s in STATE["eval"].values():
        _net = _s["network"]
        if getattr(_net, "latlon0", None) is None:
            _net.p0 = _net.p0.astype(np.float64)
            _net.p1 = _net.p1.astype(np.float64)
            _net.latlon0 = np.column_stack(
                _net.proj.to_latlon(_net.p0[:, 0], _net.p0[:, 1]))
            _net.latlon1 = np.column_stack(
                _net.proj.to_latlon(_net.p1[:, 0], _net.p1[:, 1]))
            _net.mid = (_net.p0 + _net.p1) / 2.0
            _net.mid_lat, _net.mid_lon = _net.proj.to_latlon(
                _net.mid[:, 0], _net.mid[:, 1])
            _net.length_m = _net.length_m.astype(np.float64)
            _net.bearing = _net.bearing.astype(np.float64)
        if getattr(_net, "_tree", None) is None:
            from sklearn.neighbors import BallTree
            _net._tree = BallTree(np.radians(
                np.column_stack([_net.mid_lat, _net.mid_lon])),
                metric="haversine")
    log.info("Artifacts loaded: %d evaluation trips ready", len(STATE["trips"]))


@app.get("/health")
def health():
    return {"status": "ok", "trips_loaded": len(STATE.get("trips", []))}


@app.get("/trips")
def list_trips():
    """The ten reserved evaluation trips shown in the UI."""
    log.info("GET /trips -> %d trips", len(STATE["trips"]))
    return {"trips": STATE["trips"]}


class PredictRequest(BaseModel):
    trip_id: str


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    tid = req.trip_id
    if tid not in STATE["eval"]:
        log.warning("POST /predict trip_id=%s -> unknown trip", tid)
        raise HTTPException(status_code=404, detail=f"unknown trip_id {tid!r}")

    s = STATE["eval"][tid]
    track = s["track"]
    network = s["network"]
    ref = np.asarray(s["ref_arc_idx"])

    # 1) candidate generation
    cand_lists, _ = generate_candidates(track["lat"], track["lon"], network)
    miss = candidate_miss_rate(cand_lists, ref)

    # 2) features
    rows = build_feature_rows(track, network, cand_lists, ref)

    # 3) classifier emission probabilities
    emissions = trip_emissions(STATE["model"], rows, cand_lists,
                               STATE["feature_columns"])

    # 4) classifier-only argmax and HMM/Viterbi decoding
    n = len(track)
    clf_pred = np.full(n, -1, dtype=int)
    for i, (cands, emit) in enumerate(zip(cand_lists, emissions)):
        if len(cands):
            clf_pred[i] = cands[int(np.argmax(emit))]
    hmm_pred = viterbi_decode(emissions, cand_lists, track, network, CFG)

    # Top-3 accuracy: reference arc among the 3 highest-probability candidates.
    # HMM Top-3 (honest definition): the Viterbi-decoded arc plus the two
    # next most likely candidates from the classifier emissions.
    top3_hits = top3_hmm_hits = n_valid = 0
    for i, (cands, emit) in enumerate(zip(cand_lists, emissions)):
        if ref[i] < 0 or not len(cands):
            continue
        n_valid += 1
        order = [cands[o] for o in np.argsort(emit)[::-1][:3]]
        if ref[i] in order:
            top3_hits += 1
        hmm3 = [hmm_pred[i]] + [a for a in order if a != hmm_pred[i]][:2]
        if ref[i] in hmm3:
            top3_hmm_hits += 1
    top3_acc = (top3_hits / n_valid) if n_valid else float("nan")
    top3_hmm_acc = (top3_hmm_hits / n_valid) if n_valid else float("nan")

    # 5) server-side accuracy against internal reference data
    result = {
        "trip_id": tid,
        "num_points": n,
        "candidate_miss_rate": float(miss) if miss == miss else None,
        "classifier_only_accuracy": _accuracy(clf_pred, ref),
        "classifier_top3_accuracy": float(top3_acc),
        "hmm_top3_accuracy": float(top3_hmm_acc),
        "hmm_accuracy": _accuracy(hmm_pred, ref),
        "route_continuity_hmm": route_continuity(hmm_pred, network),
        "route_continuity_classifier": route_continuity(clf_pred, network),
        "processing_time_s": time.perf_counter() - t0,
        # route for the UI: raw GPS + per-observation matched arc geometry
        "raw_points": [[float(r.lat), float(r.lon)] for r in track.itertuples()],
        "matched_arcs": [network.arc_geometry_latlon(int(a)) if a >= 0 else None
                         for a in hmm_pred],
    }
    log.info("POST /predict trip_id=%s points=%d miss_rate=%.3f "
             "acc_clf=%.3f acc_hmm=%.3f continuity=%.3f time=%.2fs",
             tid, n, miss if miss == miss else -1,
             result["classifier_only_accuracy"], result["hmm_accuracy"],
             result["route_continuity_hmm"], result["processing_time_s"])
    return result
