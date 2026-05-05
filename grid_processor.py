"""
CCMPS Video Grid Processor
===========================
Augmented from teammate's dot_twin() approach.

What this does:
- Reads the uploaded crowd video
- Uses Background Subtraction (MOG2) to detect motion blobs (people)
- Maps detections onto a 20x20 grid (same as teammate's dot_twin)
- Splits grid into 8 zones (2 rows x 4 cols) matching the CCMPS dashboard
- Serves real-time grid state via HTTP on port 5001
- Dashboard polls /api/grid every second to get live dot positions

Architecture:
  Video frames → MOG2 background subtraction → 20x20 grid occupancy
  → Zone split (8 zones) → JSON API → Dashboard Digital Twin canvas

NOTE: teammate's code uses YOLOv8 for detection. This version uses
MOG2 background subtraction as fallback (no GPU needed for POC).
When YOLOv8 is available, swap detect_people() accordingly.
"""

import cv2
import numpy as np
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── CONFIG ──────────────────────────────────────────────────────
VIDEO_PATH = "public/crowd_sample.mp4"   # path relative to CCMPS root
GRID_SIZE  = 20                          # 20x20 grid (matches teammate)
ZONE_ROWS  = 2                           # 2 rows of zones
ZONE_COLS  = 4                           # 4 cols → 8 zones total
FRAME_SKIP = 8                           # process every 8th frame (matches teammate)
PORT       = 5001                        # grid API port

# Zone names matching the dashboard
ZONE_NAMES = ['Gate A','Gate B','Main Hall','North Wing',
              'South Wing','Exit 1','Exit 2','Concourse']

# ── GLOBAL STATE ─────────────────────────────────────────────────
state = {
    "grid":        [[0]*GRID_SIZE for _ in range(GRID_SIZE)],  # 20x20
    "zones":       [],           # per-zone stats
    "total_count": 0,
    "frame_no":    0,
    "history":     [],           # crowd count over time (last 100)
    "source":      "video",
    "timestamp":   time.time(),
}
state_lock = threading.Lock()

# ── BACKGROUND SUBTRACTOR (MOG2) ─────────────────────────────────
# This is the detection engine — equivalent to teammate's tile_detect()
# but uses pixel-level motion instead of YOLO bounding boxes.
# Swap detect_people() for YOLO output if available.
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=40, detectShadows=False
)
DILATE_KERNEL = np.ones((18, 18), np.uint8)  # blob expansion

def detect_people(frame):
    """
    Detect crowd presence per pixel using MOG2.
    Returns foreground mask (same spatial layout as frame).
    In real deployment: replace with YOLOv8 bounding box centroids.
    """
    fg = fgbg.apply(frame)
    fg = cv2.dilate(fg, DILATE_KERNEL)          # fill person-sized blobs
    fg = cv2.GaussianBlur(fg, (5,5), 0)
    _, fg = cv2.threshold(fg, 128, 255, cv2.THRESH_BINARY)
    return fg

def build_grid(fg_mask, frame_shape):
    """
    Map foreground mask → 20x20 occupancy grid.
    Directly mirrors teammate's dot_twin() grid logic.
    Each cell value = occupancy ratio scaled 0-10.
    """
    h, w = frame_shape[:2]
    ch, cw = h // GRID_SIZE, w // GRID_SIZE
    grid = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            cell = fg_mask[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
            ratio = np.sum(cell > 128) / (ch * cw)
            row.append(round(ratio * 10, 2))   # 0.0 – 10.0
        grid.append(row)
    return grid

def build_zone_stats(grid):
    """
    Aggregate grid into 8 zones (2 rows x 4 cols).
    Each zone covers (GRID_SIZE/ZONE_ROWS) x (GRID_SIZE/ZONE_COLS) cells.
    """
    cells_r = GRID_SIZE // ZONE_ROWS   # 10 rows per zone
    cells_c = GRID_SIZE // ZONE_COLS   # 5 cols per zone
    zones = []
    for zr in range(ZONE_ROWS):
        for zc in range(ZONE_COLS):
            r0, r1 = zr * cells_r, (zr+1) * cells_r
            c0, c1 = zc * cells_c, (zc+1) * cells_c
            sub = [grid[r][c] for r in range(r0,r1) for c in range(c0,c1)]
            avg_occ  = sum(sub) / len(sub)
            max_occ  = max(sub)
            nonzero  = sum(1 for v in sub if v > 0.5)
            # Density proxy: average occupancy scaled to p/m²
            density  = round(avg_occ * 0.9, 2)
            congestion = min(100, int(avg_occ * 14 + max_occ * 3))
            if congestion < 30:   risk = "Safe"
            elif congestion < 55: risk = "Moderate"
            elif congestion < 75: risk = "High"
            else:                 risk = "Critical"
            idx = zr * ZONE_COLS + zc
            zones.append({
                "zone_id":         f"Z{str(idx+1).zfill(2)}",
                "name":            ZONE_NAMES[idx],
                "density":         density,
                "congestion_score": congestion,
                "risk":            risk,
                "active_cells":    nonzero,
                "grid_row_start":  r0,
                "grid_row_end":    r1,
                "grid_col_start":  c0,
                "grid_col_end":    c1,
                "source":          "video",
                "method":          "MOG2 background subtraction → 20x20 grid",
            })
    return zones

# ── SIMULATION TRANSFORMS ────────────────────────────────────────
# These mirror the 3 dashboard scenarios but applied to the REAL grid

def apply_gate_closure(grid, zones):
    """
    Gate Closure: Gates (Z01=Gate A, Z02=Gate B) blocked.
    People pile up at gate zones → density spikes there.
    Adjacent zones (Main Hall, Concourse) receive redistributed crowd.
    """
    import copy
    g = copy.deepcopy(grid)
    # Spike Gate A (top-left 10x5) and Gate B (top 10x5, cols 5-10)
    for r in range(10):
        for c in range(5):   g[r][c]   = min(10, g[r][c] * 2.2)   # Gate A
        for c in range(5,10):g[r][c]   = min(10, g[r][c] * 1.9)   # Gate B
        for c in range(10,15):g[r][c]  = min(10, g[r][c] * 1.4)   # Main Hall spillover
    explanation = (
        "Gate Closure applied: Gate A and Gate B cells show 2.2× density spike "
        "as crowd cannot exit. Main Hall receives 1.4× spillover. "
        "Recommendation: Open North Wing exits and redirect flow via Concourse."
    )
    return g, explanation

def apply_increased_inflow(grid, zones):
    """
    Increased Inflow: All entry cells get boosted.
    Simulates a large crowd entering simultaneously.
    """
    import copy
    g = copy.deepcopy(grid)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            g[r][c] = min(10, g[r][c] * 1.75 + 1.5)
    explanation = (
        "Increased Inflow applied: All zones show 1.75× density surge. "
        "Critical hotspots appear at Main Hall and Concourse (centre zones). "
        "Recommendation: Pre-deploy marshals at Z03 and Z08, reduce entry scanning rate."
    )
    return g, explanation

def apply_evacuation(grid, zones):
    """
    Emergency Evacuation: Exit zones (Z06=Exit 1, Z07=Exit 2) clear rapidly.
    Internal zones spike as crowd rushes toward exits.
    """
    import copy
    g = copy.deepcopy(grid)
    # Bottom-left (Exit 1, cols 0-5) — clear
    for r in range(10,20):
        for c in range(5):   g[r][c] = max(0, g[r][c] * 0.3)     # Exit 1 clearing
        for c in range(15,20):g[r][c] = max(0, g[r][c] * 0.3)    # Exit 2 clearing
    # Internal zones spike as people rush
    for r in range(5,15):
        for c in range(5,15): g[r][c] = min(10, g[r][c] * 2.0)   # centre rush
    explanation = (
        "Emergency Evacuation applied: Exit 1 and Exit 2 show rapid clearance (0.3×). "
        "Central zones (Main Hall, North/South Wing) spike 2× as crowd rushes to exits. "
        "Recommendation: Open ALL exits immediately, announce PA guidance, deploy at choke points."
    )
    return g, explanation

SCENARIOS = {
    "gate_closure":      apply_gate_closure,
    "increased_inflow":  apply_increased_inflow,
    "evacuation":        apply_evacuation,
}

# ── VIDEO PROCESSING THREAD ──────────────────────────────────────
def video_loop():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    frame_id  = 0
    prev_grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
    history   = []

    print(f"[GRID] Video opened: {VIDEO_PATH}")
    print(f"[GRID] Processing every {FRAME_SKIP} frames on port {PORT}")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue

        frame_id += 1
        frame = cv2.resize(frame, (1280, 720))

        if frame_id % FRAME_SKIP == 0:
            fg = detect_people(frame)
            grid = build_grid(fg, frame.shape)
            zones = build_zone_stats(grid)
            total = sum(z["active_cells"] for z in zones)

            history.append(total)
            if len(history) > 100:
                history.pop(0)

            with state_lock:
                state["grid"]        = grid
                state["zones"]       = zones
                state["total_count"] = total
                state["frame_no"]    = frame_id
                state["history"]     = history[:]
                state["timestamp"]   = time.time()

        time.sleep(0.03)  # ~30fps cadence

    cap.release()

# ── HTTP SERVER ───────────────────────────────────────────────────
class GridHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass  # silence logs

    def do_GET(self):
        if self.path == "/api/grid":
            with state_lock:
                data = {
                    "grid":        state["grid"],
                    "zones":       state["zones"],
                    "total_count": state["total_count"],
                    "frame_no":    state["frame_no"],
                    "history":     state["history"],
                    "source":      state["source"],
                    "timestamp":   state["timestamp"],
                }
            self._json(data)

        elif self.path.startswith("/api/simulate/"):
            scenario = self.path.split("/")[-1]
            if scenario in SCENARIOS:
                with state_lock:
                    base_grid = [row[:] for row in state["grid"]]
                    base_zones = state["zones"][:]
                sim_grid, explanation = SCENARIOS[scenario](base_grid, base_zones)
                sim_zones = build_zone_stats(sim_grid)
                self._json({
                    "scenario":    scenario,
                    "grid":        sim_grid,
                    "zones":       sim_zones,
                    "explanation": explanation,
                    "source":      "video+simulation",
                    "timestamp":   time.time(),
                })
            else:
                self._json({"error": "unknown scenario"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    # Start video processing in background thread
    t = threading.Thread(target=video_loop, daemon=True)
    t.start()

    # Give background subtractor time to warm up
    time.sleep(2)

    print(f"[GRID] Grid API running at http://localhost:{PORT}/api/grid")
    print(f"[GRID] Simulate: http://localhost:{PORT}/api/simulate/gate_closure")

    server = HTTPServer(("", PORT), GridHandler)
    server.serve_forever()
