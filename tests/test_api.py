"""Integration test: the shipped artifacts load and the API answers."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app, load_artifacts  # noqa: E402


@pytest.fixture(scope="session")
def client():
    load_artifacts()
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_trips_lists_ten_evaluation_trips(client):
    r = client.get("/trips")
    assert r.status_code == 200
    assert len(r.json()["trips"]) == 10


def test_predict_returns_metrics_in_range(client):
    tid = client.get("/trips").json()["trips"][0]["trip_id"]
    r = client.post("/predict", json={"trip_id": tid})
    assert r.status_code == 200
    d = r.json()
    for key in ("classifier_only_accuracy", "hmm_accuracy",
                "classifier_top3_accuracy", "hmm_top3_accuracy",
                "route_continuity_hmm"):
        assert 0.0 <= d[key] <= 1.0, key
    # the HMM top-3 set includes the decoded arc, so it can only help
    assert d["hmm_top3_accuracy"] >= 0.0
    assert len(d["raw_points"]) == d["num_points"]
    assert len(d["matched_arcs"]) == d["num_points"]


def test_predict_unknown_trip_404(client):
    r = client.post("/predict", json={"trip_id": "does-not-exist"})
    assert r.status_code == 404
