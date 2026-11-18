# Contributing

Team workflow for the PathSnap repository.

## Ground rules

- Never commit raw or processed dataset files (`data/raw/`, `data/processed/`)
  or secrets. Model weights and result CSVs/JSONs ARE committed on purpose
  (submission guidelines).
- The evaluator-test partition is quarantined: never train or tune on it.
  Tune on training/student splits, then read evaluator metrics once.

## Making a change

1. Pull latest `main`.
2. Keep commits small and focused; describe what and why in the message.
3. Run the tests before pushing:

   ```powershell
   pip install pytest
   python -m pytest tests/ -q
   ```

4. If you change pipeline code, sanity-run one decode through the API
   (see docs/api.md) and check the numbers move in the expected direction.

## Layout

- `src/` pipeline library, `api/` FastAPI backend, `ui/` evaluator UI,
  `scripts/` runnable stages, `tests/` unit + integration tests,
  `results/` experiment outputs, `report/` generated PDF.
