# Setup and data placement


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

