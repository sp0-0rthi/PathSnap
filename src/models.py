"""Model zoo: five classifiers from five algorithm families.

Every entry records its algorithm family and key configuration (the brief
requires logging configurations alongside results). Probability quality
matters because the HMM emission model consumes predicted probabilities,
so every model is wrapped to expose predict_proba.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CFG


def get_model_zoo(seed: int = CFG.seed) -> dict:
    """name -> {estimator, family, config, notes}"""
    return {
        "logistic_regression": {
            "family": "Linear / probabilistic",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]),
            "config": {"max_iter": 1000, "C": 1.0, "penalty": "l2",
                       "solver": "lbfgs", "scaling": "StandardScaler"},
        },
        "knn": {
            "family": "Instance-based",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]),
            "config": {"n_neighbors": 15, "weights": "distance",
                       "metric": "minkowski", "scaling": "StandardScaler"},
        },
        "random_forest": {
            "family": "Ensemble (bagging)",
            "estimator": RandomForestClassifier(
                n_estimators=200, random_state=seed, n_jobs=2),
            "config": {"n_estimators": 200, "max_depth": None,
                       "min_samples_split": 2, "min_samples_leaf": 1,
                       "max_features": "sqrt", "class_weight": None},
        },
        "hist_gradient_boosting": {
            "family": "Boosting",
            "estimator": HistGradientBoostingClassifier(random_state=seed),
            "config": {"max_iter": 100, "learning_rate": 0.1,
                       "max_depth": None, "loss": "log_loss"},
        },
        "mlp": {
            "family": "Neural network",
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                                      max_iter=150, early_stopping=True,
                                      random_state=seed)),
            ]),
            "config": {"hidden_layer_sizes": (64, 32), "activation": "relu",
                       "max_iter": 150, "early_stopping": True,
                       "scaling": "StandardScaler"},
        },
    }
