# Running the system (two laptops) + API reference


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


## API reference

Base URL: `http://<laptop-1-ip>:8000`

- `GET /health` -> `{"status": "ok", "trips_loaded": 10}`
- `GET /trips` -> `{"trips": [{"trip_id": "...", ...}]}` - the ten reserved
  evaluation trips.
- `POST /predict` with body `{"trip_id": "00000015"}` -> runs the full
  pipeline server-side (candidate generation -> features -> Random Forest
  emissions -> HMM/Viterbi decode) and returns:

```json
{
  "trip_id": "00000015",
  "num_points": 6854,
  "candidate_miss_rate": 0.0,
  "classifier_only_accuracy": 0.4203,
  "classifier_top3_accuracy": 0.8811,
  "hmm_accuracy": 0.3998,
  "hmm_top3_accuracy": 0.8929,
  "route_continuity_hmm": 1.0,
  "route_continuity_classifier": 1.0,
  "processing_time_s": 3.61,
  "raw_points": [[lat, lon], "..."],
  "matched_arcs": ["arc geometry per observation or null"]
}
```

`*_accuracy` fields are pointwise Top-1; `*_top3_accuracy` are accuracy@3
(reference arc within the 3 highest-scoring candidates; for the HMM variant
the decoded arc plus the next two classifier candidates). Ground truth never
leaves Laptop 1: only metrics and the predicted route geometry are returned,
and every request is logged on the server console.
