"""
server.py — FastAPI server for the CCMPS crowd analytics pipeline
================================================================
Streams test_gru.py stdout live via Server-Sent Events (SSE).

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET /api/analyze?video=<path>&width=20&height=12&horizon=30
        → SSE stream, line by line. Connect with EventSource.
          Each event: data: {"line": "..."}
          Final event: data: {"done": true, "exit_code": 0}

    GET /api/results
        → risk_log.csv as JSON array (call after /api/analyze finishes)

    GET /api/health
        → {"status": "ok"}

Video param:
    - Absolute path  → used directly (for videos uploaded to the Express repo)
    - Filename only  → resolved inside CCMPS/crowd_analytics/ folder
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
SCRIPT_DIR = Path(__file__).parent / "CCMPS" / "crowd_analytics"
CSV_PATH   = SCRIPT_DIR / "risk_log.csv"

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CCMPS Crowd Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ── ANALYZE — SSE stream ──────────────────────────────────────────────────────
@app.get("/api/analyze")
def analyze(
    video:   str   = Query(default="Crowd.mp4"),
    width:   float = Query(default=20.0),
    height:  float = Query(default=12.0),
    horizon: int   = Query(default=30),
):
    # Accept either absolute path or filename relative to crowd_analytics/
    video_path = video if Path(video).is_absolute() else str(SCRIPT_DIR / video)

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
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(SCRIPT_DIR),
        )
        for raw in proc.stdout:
            line = raw.rstrip()
            if line:
                yield f"data: {json.dumps({'line': line})}\n\n"
        proc.wait()
        yield f"data: {json.dumps({'done': True, 'exit_code': proc.returncode})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── RESULTS ───────────────────────────────────────────────────────────────────
@app.get("/api/results")
def get_results():
    if not CSV_PATH.exists():
        return JSONResponse({"error": "No results yet."}, status_code=404)
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
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
    return {"status": "ok", "script_dir": str(SCRIPT_DIR)}
