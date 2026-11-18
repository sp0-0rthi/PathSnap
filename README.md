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

## Setup (Windows / PowerShell, VS Code)

1. Install Python 3.10+ from python.org (tick "Add python.exe to PATH").
2. Open the project folder in VS Code (File > Open Folder).
3. Open a PowerShell terminal in VS Code (Terminal > New Terminal) and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# if activation is blocked: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
pip install -r requirements.txt
```

## Data placement

Put the dataset under `data/raw/`, one folder per trip, each containing
`<trip>.track`, `<trip>.nodes`, `<trip>.arcs` and (where supplied)
`<trip>.route` - the loader discovers them automatically. If the dataset
ships its own partition folders (`train/`, `student_test/`,
`evaluator_test/`) they are honoured; otherwise a deterministic seeded
80/10/10 trip split is used.

No dataset handy? Generate a stand-in in the exact same format:

```powershell
python scripts\make_synthetic_data.py --trips 100
```

## Rerun on the real dataset (one command)

With the real trips placed under `data/raw/` (folders `train/`,
`student_test/`, `evaluator_test/` - or any flat layout, the split falls
back to a seeded 80/10/10), the full rebuild - EDA, experiments, HPO, HMM
evaluation, screenshots, demo, report PDF - is one PowerShell line:

```powershell
python scripts\run_eda.py; python scripts\run_pipeline.py --force-rebuild; python scripts\make_screenshots.py; python scripts\make_standalone_demo.py; python scripts\build_report.py
```

`--force-rebuild` clears every cached artifact, including the persisted
HPO study, so no synthetic-data results leak into the real run.

## Run the experiments

```powershell
python scripts\run_eda.py          # dataset understanding & EDA outputs
python scripts\run_pipeline.py     # model comparison -> HPO -> HMM -> artifacts
python scripts\build_report.py     # regenerate the technical report PDF
```

`run_pipeline.py` writes all metrics to `results/` (model_comparison.csv,
hpo_summary.json, hpo_trials.csv, hmm_evaluation.json, experiment_log.json,
figures/, maps/) and exports the serving artifacts to `artifacts/`.

## Run the system (two laptops)

Laptop 1 (model + API):

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Laptop 2 (UI): open `ui/index.html` in a browser (double-click), set the API
base URL to Laptop 1's address (e.g. `http://192.168.1.10:8000`), click
"Load evaluation trips", pick any of the ten trips, click "Reconstruct
route". The map shows the raw GPS points and the reconstructed route; the
panel shows accuracy (classifier and HMM), route continuity and the
candidate miss rate, all computed server-side.

Accuracy (accuracy@3): fraction of points where the true road is within the
model's top 3 predictions. (Pointwise Top-1 accuracy is reported separately
in the results tables below.) Ground truth never leaves
Laptop 1; every request is logged on Laptop 1's console.

`ui/demo_standalone.html` is a server-free snapshot with real API responses
embedded - double-click to inspect the UI behaviour.

## Report screenshots

Report figures under `results/figures/` include the UI screenshot and
terminal captures of successful API responses, captured for real by
`scripts/make_screenshots.py`. To re-capture them on the presentation
laptops (if evaluators require it): start the API on Laptop 1, open the UI
on Laptop 2, run a prediction, and screenshot the UI plus Laptop 1's
console, or simply rerun `python scripts\make_screenshots.py` there.

## Notes

* No secrets are stored anywhere in this repository.
* **Trained weights:** `artifacts/random_forest_tuned.joblib` is the
  dedicated model-weights file the API loads at startup; see
  `artifacts/MODEL_WEIGHTS.md` for the tuned hyperparameters and the
  metrics it achieved. Configuration (`artifacts/hmm_config.json`) is
  committed alongside, as required by the submission guidelines.
* Every metric in the report PDF is regenerated from the committed outputs
  under `results/` by `scripts/build_report.py`.

## v3.1 - persistence blend (shipped during the live eval)

The Viterbi decoder now adds a stay bonus (+1.5 log units) to same-arc
transitions (`stay_bonus` in src/config.py; set to 0.0 to reproduce v3).
Tuned on student validation trips only; evaluator_test was read once, after
tuning.

| metric (HMM) | student_test v3 | student_test v3.1 | evaluator_test v3.1 |
|---|---|---|---|
| Top-1 pointwise | 51.3% | **57.6%** | 51.0% |
| Top-1 mean/trip | 55.4% | **59.7%** | 53.4% |
| accuracy mean/trip | 91.6% | **93.2%** | 91.8% |
| route continuity | 99.99% | 99.99% | 100% |

Full numbers: results/blend_evaluation.json.
