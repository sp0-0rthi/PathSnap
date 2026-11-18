# Running the experiments and evaluation


```powershell
python scripts\run_eda.py          # dataset understanding & EDA outputs
python scripts\run_pipeline.py     # model comparison -> HPO -> HMM -> artifacts
python scripts\build_report.py     # regenerate the technical report PDF
```

`run_pipeline.py` writes all metrics to `results/` (model_comparison.csv,
hpo_summary.json, hpo_trials.csv, hmm_evaluation.json, experiment_log.json,
figures/, maps/) and exports the serving artifacts to `artifacts/`.



Report figures under `results/figures/` include the UI screenshot and
terminal captures of successful API responses, captured for real by
`scripts/make_screenshots.py`. To re-capture them on the presentation
laptops (if evaluators require it): start the API on Laptop 1, open the UI
on Laptop 2, run a prediction, and screenshot the UI plus Laptop 1's
console, or simply rerun `python scripts\make_screenshots.py` there.



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
