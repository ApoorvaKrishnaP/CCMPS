"""
server.py — FastAPI server for the CCMPS crowd analytics pipeline
================================================================
Exposes test_gru.py as an HTTP API with live SSE streaming.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET /api/analyze  — SSE stream of test_gru.py stdout (use with EventSource)
    GET /api/results  — Full risk_log.csv as JSON (call after analysis)
    GET /api/health   — Health check
"""

import json
import subprocess
import sys
import csv
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── PATHS ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent   # server.py lives inside crowd_analytics/
CSV_PATH   = SCRIPT_DIR / "risk_log.csv"

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CCMPS Crowd Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Express origin in production
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── ANALYZE — SSE stream of test_gru.py stdout ────────────────────────────────
@app.get("/api/analyze")
def analyze(
    video:   str   = Query(default="Crowd.mp4", description="Video filename inside crowd_analytics/"),
    width:   float = Query(default=20.0,  description="Zone width in metres"),
    height:  float = Query(default=12.0,  description="Zone height in metres"),
    horizon: int   = Query(default=30,    description="Forecast horizon in seconds"),
):
    """
    Server-Sent Events endpoint.
    Connect with:  new EventSource('/api/stream?video=Crowd.mp4')  (via Express proxy)
    Each event:    data: {"line": "<stdout line>"}
    Final event:   data: {"done": true, "exit_code": 0}
    """
    video_path = str(SCRIPT_DIR / video) if not Path(video).is_absolute() else video

    def generate():
        proc = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT_DIR / "test_gru.py"),
                "--video",   video_path,
                "--width",   str(width),
                "--height",  str(height),
                "--horizon", str(horizon),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,          # line-buffered
            cwd=str(SCRIPT_DIR),
        )

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            if line:
                yield f"data: {json.dumps({'line': line})}\n\n"

        proc.wait()
        yield f"data: {json.dumps({'done': True, 'exit_code': proc.returncode})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering if behind a proxy
        },
    )


# ── RESULTS — full CSV as JSON ────────────────────────────────────────────────
@app.get("/api/results")
def get_results():
    """Returns risk_log.csv rows as a JSON array. Call after /api/analyze finishes."""
    if not CSV_PATH.exists():
        return JSONResponse(
            {"error": "No results yet — run /api/analyze first."},
            status_code=404,
        )

    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "second":         int(row["second"]),
                "people":         float(row["people"]),
                "density":        float(row["density"]),
                "speed":          float(row["speed"]),
                "stagnation":     float(row["stagnation"]),
                "raw_label":      row["raw_label"],
                "confidence":     float(row["confidence"]),
                "confirmed_risk": row["confirmed_risk"],
            })

    return JSONResponse(rows)


# ── HEALTH ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status":     "ok",
        "script_dir": str(SCRIPT_DIR),
        "csv_exists": CSV_PATH.exists(),
    }
