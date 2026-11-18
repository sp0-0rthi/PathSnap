# Trained model weights

* **File:** `artifacts/random_forest_tuned.joblib` (scikit-learn
  RandomForestClassifier, loaded by the API at startup)
* **Model:** Random Forest, selected from a five-family comparison
  (Logistic Regression, KNN, Random Forest, HistGradientBoosting,
  MLPClassifier) and tuned with Bayesian HPO (Optuna TPE,
  4 trials, objective: mean GroupKFold(5) log loss on 80 training trips (minimise))
* **Features:** ['dist_to_segment', 'heading_diff', 'speed']

## Tuned hyperparameters

```json
{
  "n_estimators": 250,
  "max_depth": 31,
  "min_samples_split": 15,
  "min_samples_leaf": 6,
  "max_features": "sqrt",
  "class_weight": "none"
}
```

## Achieved metrics (student-testing partition)

* Top-1 accuracy: 0.5203
* Macro F1: 0.6198
* Precision / recall: 0.5316 / 0.2033
* Log loss: 0.2178
* Route continuity (classifier-only): 0.9883
* Best CV log loss during HPO: 0.2132

Full experiment outputs: `results/model_comparison.csv`,
`results/hpo_summary.json`, `results/hpo_trials.csv`,
`results/hmm_evaluation.json`.
