"""Build a standalone, double-clickable demo of the PathSnap UI.

ui/demo_standalone.html embeds REAL API responses (trips list + one
prediction per evaluation trip) captured from the live backend, so anyone
can open the file in a browser - no server needed - and inspect the UI
with the predicted route and metrics for each evaluation trip.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def wait_for(url, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"{url} never came up")


def main():
    logf = open("/tmp/ps_demo_api.log", "w")
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app",
                            "--host", "127.0.0.1", "--port", "8000"],
                           cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                           start_new_session=True)
    try:
        wait_for("http://127.0.0.1:8000/health")
        trips = json.loads(subprocess.run(
            ["curl", "-s", "http://127.0.0.1:8000/trips"],
            capture_output=True, text=True).stdout)["trips"]
        # The full set is served by the API; the embedded demo caps at the
        # first three trips to keep the file self-contained and quick to
        # regenerate (each prediction is decoded live by the backend).
        trips = trips[:3]
        preds = {}
        for t in trips:
            out = subprocess.run(
                ["curl", "-s", "-X", "POST", "http://127.0.0.1:8000/predict",
                 "-H", "Content-Type: application/json",
                 "-d", json.dumps({"trip_id": t["trip_id"]})],
                capture_output=True, text=True).stdout
            preds[t["trip_id"]] = json.loads(out)
    finally:
        api.terminate()
        logf.close()

    page = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PathSnap - Standalone UI Demo</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body { font-family: Arial, sans-serif; margin: 0; display: flex; height: 100vh; }
  #panel { width: 320px; padding: 16px; box-sizing: border-box; background: #f7f7f7;
           border-right: 1px solid #ccc; overflow-y: auto; }
  #map { flex: 1; }
  h1 { font-size: 20px; margin: 0 0 12px; }
  select { width: 100%; padding: 7px; margin-top: 8px; }
  #metrics { margin-top: 16px; font-size: 13px; background: #fff; padding: 10px;
             border: 1px solid #ddd; border-radius: 6px; }
  .row { display: flex; justify-content: space-between; margin: 3px 0; }
  .note { font-size: 12px; color: #666; margin-top: 10px; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; }
</style>
</head>
<body>
<div id="panel">
  <h1>PathSnap</h1>
  <div>Standalone demo - real API responses embedded, no server needed.</div>
  <select id="tripSelect"></select>
  <div id="metrics"></div>
  <div class="note">
    <span class="dot" style="background:#1f77b4"></span> Raw GPS observations<br>
    <span class="dot" style="background:#d62728"></span> Predicted route (HMM/Viterbi)
  </div>
</div>
<div id="map"></div>
<script>
const DATA = __DATA__;
const map = L.map('map').setView([12.935, 77.605], 14);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            {maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(map);
let rawLayer = null, predLayer = null;
const sel = document.getElementById('tripSelect');
Object.keys(DATA).forEach(t => {
  const o = document.createElement('option');
  o.value = t; o.textContent = `${t} (${DATA[t].num_points} points)`;
  sel.appendChild(o);
});
function show(tid) {
  const d = DATA[tid];
  if (rawLayer) map.removeLayer(rawLayer);
  if (predLayer) map.removeLayer(predLayer);
  rawLayer = L.layerGroup(d.raw_points.map(p =>
    L.circleMarker(p, {radius: 2.5, color: '#1f77b4', fillOpacity: 0.6}))).addTo(map);
  predLayer = L.layerGroup();
  d.matched_arcs.forEach(g => { if (g) L.polyline(g, {color: '#d62728', weight: 4, opacity: 0.85}).addTo(predLayer); });
  predLayer.addTo(map);
  if (d.raw_points.length) map.fitBounds(L.latLngBounds(d.raw_points));
  document.getElementById('metrics').innerHTML = `
    <b>${d.trip_id}</b>
    <div class="row"><span>Observations</span><span>${d.num_points}</span></div>
    <div class="row"><span>Top-1 accuracy (classifier)</span><span>${(100*d.classifier_only_accuracy).toFixed(1)}%</span></div>
    <div class="row"><span><b>Top-1 accuracy (HMM)</b></span><span><b>${(100*d.hmm_accuracy).toFixed(1)}%</b></span></div>
    <div class="row"><span>Route continuity (HMM)</span><span>${(100*d.route_continuity_hmm).toFixed(1)}%</span></div>
    <div class="row"><span>Candidate miss rate</span><span>${(100*d.candidate_miss_rate).toFixed(1)}%</span></div>`;
}
sel.onchange = () => show(sel.value);
show(sel.value);
</script>
</body>
</html>
"""
    page = page.replace("__DATA__", json.dumps(preds))
    out = ROOT / "ui" / "demo_standalone.html"
    out.write_text(page)
    print("wrote", out)


if __name__ == "__main__":
    main()
