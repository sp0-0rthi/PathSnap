<!-- PathSnap v3.1 - docs (owner: Spoorthi) -->
# PathSnap - Distributed Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University.

PathSnap reconstructs realistic vehicle trajectories by matching noisy GPS
observations to a road network: candidate generation with a KDTree/BallTree
over arc midpoints, candidate classification with a tuned Random Forest, and
route reconstruction with an HMM/Viterbi decoder over the road-network
connectivity. A FastAPI backend (Laptop 1) serves the ten reserved evaluation
trips to an evaluator UI (Laptop 2).

## Team

- Spoorthi A N (sp0-0rthi) - data pipeline, features, documentation
- Shubharth (shubharth2006) - API, Root Decoder
- Bindu (Bindusuresh) - evaluator UI and demo
- Sourabh (SourabhBPatil) - model zoo, training, HPO

## Repository layout

```
pathsnap/
├── data/
│   ├── raw/                 # dataset: one folder per trip (.track/.nodes/.arcs/.route)
│   └── processed/           # cached feature tables, partitions (generated)
├── src/                     # all pipeline source code
│   ├── config.py            # paths, constants, seeds
│   ├── geo.py               # haversine, local projection, geometry
│   ├── data_processing.py   # raw loaders, validation, cleaning
│   ├── labeling.py          # GPS -> reference arc alignment (monotonic DP)
│   ├── road_network.py      # network geometry, spatial index, connectivity
│   ├── candidates.py        # candidate generation + Candidate Miss Rate
│   ├── features.py          # dist_to_segment, heading_diff, speed
│   ├── splits.py            # 80/10/10 trip partitions (leakage-safe)
│   ├── models.py            # the five-model zoo (5 families)
│   ├── train_models.py      # model comparison experiments
│   ├── hpo.py               # Bayesian HPO (Optuna TPE) of the Random Forest
│   ├── hmm.py               # HMM / Viterbi route reconstruction
│   ├── evaluate.py          # classifier-only vs classifier+HMM
│   ├── metrics.py           # shared metrics (Top-1, log loss, continuity)
│   ├── visualize.py         # figures + Folium maps
│   └── export_artifacts.py  # model/config/eval bundles for the API
├── scripts/
│   ├── make_synthetic_data.py   # stand-in dataset in the exact raw format
│   ├── run_eda.py               # dataset understanding & EDA outputs
│   ├── run_pipeline.py          # full experiment pipeline (one command)
│   ├── build_report.py          # generates report/PathSnap_Technical_Report.pdf
│   ├── make_screenshots.py      # captures real UI/API screenshots
│   └── make_standalone_demo.py  # builds ui/demo_standalone.html
├── api/main.py              # FastAPI backend (Laptop 1, port 8000)
├── ui/index.html            # evaluator UI (Laptop 2)
├── ui/demo_standalone.html  # double-clickable demo (embedded real responses)
├── artifacts/               # trained model, HMM config, eval bundles (generated)
├── results/                 # CSVs, logs, figures, maps (generated)
├── report/                  # generated technical report PDF
└── requirements.txt
```


## Documentation

- [docs/setup.md](docs/setup.md) - installation, data placement, rerunning the pipeline
- [docs/api.md](docs/api.md) - two-laptop run + API reference
- [docs/evaluation.md](docs/evaluation.md) - experiments, metrics definitions, v3.1 results

## Notes

* No secrets are stored anywhere in this repository.
* **Trained weights:** `artifacts/random_forest_tuned.joblib` is the
  dedicated model-weights file the API loads at startup; see
  `artifacts/MODEL_WEIGHTS.md` for the tuned hyperparameters and the
  metrics it achieved. Configuration (`artifacts/hmm_config.json`) is
  committed alongside, as required by the submission guidelines.
* Every metric in the report PDF is regenerated from the committed outputs
  under `results/` by `scripts/build_report.py`.

