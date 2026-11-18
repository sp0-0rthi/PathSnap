"""Capture real UI and API screenshots for the technical report.

Starts the FastAPI server and a static server for the UI, drives the UI
with headless Chromium (real clicks, real API responses), and captures:
* results/figures/ui_screenshot.png        - the working UI after a real
                                             prediction
* results/figures/api_terminal_trips.png   - terminal-styled capture of a
                                             real `curl /trips` session
* results/figures/api_terminal_predict.png - same for `curl -X POST
                                             /predict`
* results/figures/api_server_log.png       - the API server's own request
                                             log (what evaluators ask to
                                             see on the receiving laptop)

These were captured on the build machine; teams can re-capture them on
their own laptops (see README, section "Report screenshots").
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def wait_for(url: str, timeout: float = 30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"{url} never came up")


def terminal_png(title: str, lines: list[str], out: Path):
    """Render real captured terminal text as a terminal-styled PNG."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
        bold = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        bold = font
    pad, lh = 18, 22
    width = 1180
    height = pad * 2 + 34 + lh * len(lines)
    img = Image.new("RGB", (width, height), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, 34], fill=(60, 60, 60))
    d.ellipse([12, 10, 26, 24], fill=(255, 95, 86))
    d.ellipse([32, 10, 46, 24], fill=(255, 189, 46))
    d.ellipse([52, 10, 66, 24], fill=(39, 201, 63))
    d.text((80, 8), title, fill=(230, 230, 230), font=bold)
    y = 34 + pad
    for line in lines:
        if line.startswith("$"):
            d.text((pad, y), line, fill=(140, 220, 140), font=bold)
        else:
            d.text((pad, y), line, fill=(220, 220, 220), font=font)
        y += lh
    img.save(out)
    return out


def wrap(text: str, width: int = 130):
    out = []
    for line in text.splitlines():
        while len(line) > width:
            out.append(line[:width])
            line = "    " + line[width:]
        out.append(line)
    return out


def main():
    procs = []

    def start(cmd, logfile):
        f = open(logfile, "w")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((p, f))
        return p

    api = start([sys.executable, "-m", "uvicorn", "api.main:app",
                 "--host", "127.0.0.1", "--port", "8000"], "/tmp/ps_api.log")
    ui = start([sys.executable, "-m", "http.server", "8501", "--directory",
                str(ROOT / "ui")], "/tmp/ps_ui.log")
    try:
        wait_for("http://127.0.0.1:8000/health")
        wait_for("http://127.0.0.1:8501/index.html")

        # ---- real curl sessions ----
        trips_out = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:8000/trips"],
            capture_output=True, text=True).stdout
        trips = json.loads(trips_out)
        tid = trips["trips"][0]["trip_id"]
        pred_out = subprocess.run(
            ["curl", "-s", "-X", "POST", "http://127.0.0.1:8000/predict",
             "-H", "Content-Type: application/json",
             "-d", json.dumps({"trip_id": tid})],
            capture_output=True, text=True).stdout
        pred = json.loads(pred_out)
        pred_show = {k: (f"<{len(v)} values>" if isinstance(v, list) else v)
                     for k, v in pred.items()}

        terminal_png(
            "Laptop 1 - GET /trips",
            ["$ curl http://localhost:8000/trips", ""]
            + wrap(json.dumps(json.loads(trips_out), indent=2)),
            FIG / "api_terminal_trips.png")
        terminal_png(
            "Laptop 1 - POST /predict",
            [f"$ curl -X POST http://localhost:8000/predict -H \"Content-Type: application/json\" -d '{{\"trip_id\": \"{tid}\"}}'", ""]
            + wrap(json.dumps(pred_show, indent=2)),
            FIG / "api_terminal_predict.png")

        # ---- UI screenshot with a real prediction ----
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.goto("http://127.0.0.1:8501/index.html")
            page.click("#loadBtn")
            page.wait_for_selector("#tripSelect:not([disabled])", timeout=15000)
            page.select_option("#tripSelect", tid)
            page.click("#predictBtn")
            page.wait_for_function(
                "document.getElementById('metrics').style.display === 'block'",
                timeout=240000)
            page.wait_for_timeout(3500)  # let map tiles settle
            page.screenshot(path=str(FIG / "ui_screenshot.png"))
            browser.close()

        # ---- server log capture ----
        time.sleep(1)
        log_lines = Path("/tmp/ps_api.log").read_text().splitlines()
        interesting = [l for l in log_lines if l.strip()][-14:]
        terminal_png("Laptop 1 - PathSnap API server log", interesting,
                     FIG / "api_server_log.png")
        print("screenshots written to", FIG)
    finally:
        for p, f in procs:
            p.terminate()
            f.close()


if __name__ == "__main__":
    main()
