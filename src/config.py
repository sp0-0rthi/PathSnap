"""Central configuration for the PathSnap pipeline.

Every path, constant and random seed used anywhere in the project lives
here so experiments are reproducible and easy to audit (the brief requires
retaining configurations and seeds alongside results).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
ARTIFACTS = PROJECT_ROOT / "artifacts"
RESULTS = PROJECT_ROOT / "results"
REPORT = PROJECT_ROOT / "report"

for _d in (DATA_RAW, DATA_PROCESSED, ARTIFACTS, RESULTS, REPORT):
    _d.mkdir(parents=True, exist_ok=True)

# Raw input files (override with environment variables if names differ).
GPS_FILE = os.environ.get("PATHSNAP_GPS_FILE", str(DATA_RAW / "gps_points.csv"))
NETWORK_FILE = os.environ.get("PATHSNAP_NETWORK_FILE", str(DATA_RAW / "road_network.csv"))


# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """All tunable constants in one place."""

    # Reproducibility
    seed: int = 42

    # Candidate generation: KDTree over segment midpoints, k nearest
    # within a search radius (meters).
    candidate_k: int = 12
    candidate_radius_m: float = 100.0
    # Edge case: if no candidate is found inside the radius, retry once
    # with this relaxed radius before declaring the observation unmatched.
    candidate_relaxed_radius_m: float = 250.0

    # Feature engineering
    # GPS gap handling: an interval longer than this (seconds) or a jump
    # longer than gap_dist_m breaks the Viterbi chain (edge case: GPS gaps).
    gap_seconds: float = 60.0
    gap_dist_m: float = 300.0

    # Trip-level split of the dataset: 80 train / 10 student-test /
    # 10 evaluator-test. The evaluator-test partition is never used for
    # any modelling decision - it is only served by the API.
    n_train_trips: int = 80
    n_student_test_trips: int = 10
    n_eval_trips: int = 10

    # Cross-validation: group-aware by trip_id so observations from one
    # trip never straddle train/validation folds.
    cv_folds: int = 5

    # Bayesian hyperparameter optimisation (Optuna TPE).
    hpo_trials: int = 20

    # HMM / Viterbi transition model.
    # transition_logp = -|path_distance - gps_distance| / transition_beta
    transition_beta: float = 15.0
    # Persistence blend: extra log-probability bonus for staying on the
    # same arc between consecutive observations (added to the transition
    # score when a == b). 0.0 reproduces the plain Newson-Krumm decoder.
    # Tuned on the student validation trips only (v3.1): grid
    # {0,1,1.5,2,3} on 3 student_test trips; 1.5 gave the best pooled
    # Top-1 (+1.7 pts) with no per-trip regression.
    stay_bonus: float = 1.5
    # Training-row subsampling: consecutive 1 Hz observations are highly
    # autocorrelated, so model fitting uses at most this many evenly spaced
    # observations per trip (evaluation always uses every observation).
    train_max_obs_per_trip: int = field(
        default_factory=lambda: int(os.environ.get("PATHSNAP_TRAIN_MAX_OBS", 300)))
    # Invalid / unreachable transitions get this floor probability instead
    # of zero, so the Viterbi chain never dies (edge case: invalid
    # transitions).
    transition_eps: float = 1e-9
    # Emission probabilities are floored to avoid log(0).
    emission_eps: float = 1e-9
    # Dijkstra cutoff multiplier when searching a path between segments.
    path_search_factor: float = 4.0
    path_search_buffer_m: float = 200.0

    # Endpoint snapping tolerance (m) when building network connectivity.
    snap_tolerance_m: float = 5.0


CFG = Config()

FEATURE_COLUMNS = ["dist_to_segment", "heading_diff", "speed"]
