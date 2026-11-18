"""PathSnap - Distributed Map-Matching & Route Reconstruction.

End-to-end pipeline:
    raw GPS + road network -> candidate generation (KDTree over segment
    midpoints) -> feature engineering -> model comparison (5 algorithm
    families) -> Bayesian HPO (Optuna TPE) -> HMM/Viterbi route
    reconstruction -> FastAPI backend -> evaluator UI.
"""

__version__ = "1.0.0"
