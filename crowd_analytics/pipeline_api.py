"""
CCMPS Pipeline API Server
==========================
Wraps the YOLO+DeepSORT crowd analytics pipeline as a lightweight HTTP API.
Spawned on-demand by server.js when a user uploads a video.

Port : 5002
Endpoints:
  GET /metrics  -> latest { density, avg_speed, inflow, outflow, congestion_score, risk, ... }
  GET /status   -> { status: "loading" | "running" | "stopped" | "error" }
  GET /stop     -> gracefully stop the processing thread

Usage (called by server.js):
  python crowd_analytics/pipeline_api.py
        --video  <absolute_path_to_video>
        --zone   "North Wing"
        --width  20.0
        --height 12.0
"""

import os, sys, argparse, threading, time, json, math
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler

# Force UTF-8 output so Windows cp1252 console never crashes on Unicode
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── CLI ARGS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--video",  required=True,  help="Absolute path to video file")
parser.add_argument("--zone",   default="North Wing", help="Zone name")
parser.add_argument("--width",  type=float, default=20.0, help="Zone real-world width (m)")
parser.add_argument("--height", type=float, default=12.0, help="Zone real-world height (m)")
parser.add_argument("--port",   type=int,   default=5002)
args = parser.parse_args()

# Model lives next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "yolov8n.pt")

# ── PIPELINE CONFIG ───────────────────────────────────────────────────────────
CONF               = 0.25
IOU                = 0.45
TILE_SIZE          = 640
OVERLAP            = 0.2
PROCESS_EVERY      = 5          # run YOLO every N frames
COUNT_LINE_RATIO   = 0.5        # virtual counting line at 50% frame height

# ── SHARED STATE (written by pipeline thread, read by HTTP handler) ───────────
state = {
    "zone":             args.zone,
    "density":          0.0,
    "avg_speed":        0.0,
    "inflow":           0.0,
    "outflow":          0.0,
    "people_count":     0,
    "frame_no":         0,
    "congestion_score": 0,
    "risk":             "Safe",
    "status":           "loading",
    "source":           "pipeline",
    "timestamp":        time.time(),
}
state_lock  = threading.Lock()
stop_event  = threading.Event()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def nms_opencv(boxes, scores, iou_thr=0.4):
    if not boxes:
        return [], []
    xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
    idx  = cv2.dnn.NMSBoxes(xywh, scores, 0.0, iou_thr)
    if len(idx) == 0:
        return [], []
    flat = idx.flatten()
    return [boxes[i] for i in flat], [scores[i] for i in flat]


def tile_detect(frame, model):
    """Run tiled YOLOv8 detection, return merged (boxes, scores)."""
    h, w = frame.shape[:2]
    step = int(TILE_SIZE * (1 - OVERLAP))
    boxes_all, scores_all = [], []
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = frame[y:y + TILE_SIZE, x:x + TILE_SIZE]
            if tile.shape[0] < 100 or tile.shape[1] < 100:
                continue
            results = model(tile, conf=CONF, iou=IOU, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    if int(box.cls[0]) != 0:   # class 0 = person
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes_all.append([x1 + x, y1 + y, x2 + x, y2 + y])
                    scores_all.append(float(box.conf[0]))
    return boxes_all, scores_all


def compute_congestion(density, avg_speed, inflow, outflow):
    """Same multi-factor formula as server.js so scores are comparable."""
    d_comp    = min(100.0, (density / 9.0) * 100.0)
    spd_pen   = max(0.0,   (1.0 - avg_speed / 2.0) * 20.0)
    net_flow  = max(0.0, inflow - outflow)
    flow_pres = min(25.0, (net_flow / 150.0) * 25.0)
    score     = int(min(100, d_comp + spd_pen + flow_pres))
    if   score < 30: risk = "Safe"
    elif score < 55: risk = "Moderate"
    elif score < 75: risk = "High"
    else:            risk = "Critical"
    return score, risk

# ── PIPELINE THREAD ───────────────────────────────────────────────────────────
def pipeline_loop():
    # Deferred import so HTTP server can start while model loads
    try:
        from ultralytics import YOLO
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError as e:
        with state_lock:
            state["status"] = f"error: missing dependency — {e}"
        print(f"[PIPELINE] Import error: {e}", flush=True)
        return

    print(f"[PIPELINE] Loading YOLO from {MODEL_PATH}", flush=True)
    try:
        model   = YOLO(MODEL_PATH)
        tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    except Exception as e:
        with state_lock:
            state["status"] = f"error: {e}"
        print(f"[PIPELINE] Model load error: {e}", flush=True)
        return

    area_m2 = args.width * args.height
    cap     = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        with state_lock:
            state["status"] = "error: cannot open video"
        print(f"[PIPELINE] Cannot open video: {args.video}", flush=True)
        return

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count_y  = int(frame_h * COUNT_LINE_RATIO)
    ppm      = ((frame_w / args.width) + (frame_h / args.height)) / 2  # pixels per metre

    prev_pos       = {}   # track_id -> (cx, cy)
    track_side     = {}   # track_id -> bool (above/below line)
    in_count = out_count = 0
    start_time = time.time()
    frame_id   = 0

    with state_lock:
        state["status"] = "running"

    print(f"[PIPELINE] Running — zone={args.zone} area={area_m2:.1f}m² fps={fps}", flush=True)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
            continue

        frame_id += 1
        if frame_id % PROCESS_EVERY != 0:
            continue

        # ── Detect ──────────────────────────────────────────────────────────
        boxes, scores = tile_detect(frame, model)
        final_boxes, final_scores = nms_opencv(boxes, scores)

        detections = [
            ([x1, y1, x2 - x1, y2 - y1], s, "person")
            for (x1, y1, x2, y2), s in zip(final_boxes, final_scores)
        ]
        tracks = tracker.update_tracks(detections, frame=frame)

        # ── Track metrics ────────────────────────────────────────────────────
        people_count = 0
        speeds       = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid       = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy    = (l + r) // 2, (t + b) // 2
            people_count += 1

            # Speed
            if tid in prev_pos:
                px, py  = prev_pos[tid]
                dist_m  = math.sqrt((cx - px) ** 2 + (cy - py) ** 2) / ppm
                spd     = dist_m / (PROCESS_EVERY / fps)
                speeds.append(spd)
            prev_pos[tid] = (cx, cy)

            # Inflow / Outflow across virtual line
            cur_side = cy > count_y
            if tid not in track_side:
                track_side[tid] = cur_side
            if track_side[tid] != cur_side:
                if cur_side:  in_count  += 1
                else:         out_count += 1
                track_side[tid] = cur_side

        # ── Compute aggregates ───────────────────────────────────────────────
        elapsed_min = max((time.time() - start_time) / 60.0, 1 / 60.0)
        density     = people_count / area_m2
        avg_speed   = float(np.mean(speeds)) if speeds else 0.0
        inflow      = in_count  / elapsed_min
        outflow     = out_count / elapsed_min
        score, risk = compute_congestion(density, avg_speed, inflow, outflow)

        with state_lock:
            state.update({
                "density":          round(density,   3),
                "avg_speed":        round(avg_speed, 3),
                "inflow":           round(inflow,    2),
                "outflow":          round(outflow,   2),
                "people_count":     people_count,
                "frame_no":         frame_id,
                "congestion_score": score,
                "risk":             risk,
                "timestamp":        time.time(),
            })

        print(
            f"[PIPELINE] f{frame_id:05d} | people={people_count} "
            f"density={density:.3f} speed={avg_speed:.2f} "
            f"in={inflow:.1f} out={outflow:.1f} score={score} {risk}",
            flush=True
        )

    cap.release()
    with state_lock:
        state["status"] = "stopped"
    print("[PIPELINE] Stopped.", flush=True)

# ── HTTP HANDLER ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default access log

    def do_GET(self):
        if self.path == "/metrics":
            with state_lock:
                data = dict(state)
            self._json(data)

        elif self.path == "/stop":
            stop_event.set()
            self._json({"status": "stopping"})

        elif self.path == "/status":
            with state_lock:
                s = state["status"]
            self._json({"status": s})

        else:
            self._json({"error": "not found"}, 404)

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type",  "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start pipeline in background — HTTP server responds immediately
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()

    print(f"[PIPELINE] HTTP API -> http://localhost:{args.port}", flush=True)
    print(f"[PIPELINE] Zone: '{args.zone}' | {args.width}m x {args.height}m", flush=True)

    server = HTTPServer(("", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_event.set()
