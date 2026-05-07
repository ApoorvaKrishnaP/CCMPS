"""
vid_twin_api.py — Headless HTTP API that replicates the logic of vid_twin_map.py
and serves four live MJPEG streams (video, zone_twin, dot_twin, graph) to the
CCMPS frontend Digital-Twin Live Analytics subsection.

Streams:
  GET /stream/video  ->  MJPEG — split-view detection video (6 zones)
  GET /stream/zone   ->  MJPEG — zone twin grid (6 coloured cells)
  GET /stream/dot    ->  MJPEG — 20x20 dot density grid
  GET /stream/graph  ->  MJPEG — crowd count history graph

Status:
  GET  /status       ->  JSON { status, frame_id, total_count, error }
  POST /stop         ->  request graceful shutdown

Usage (spawned by server.js):
    python vid_twin_api.py --video <path> --port <port>
"""

import argparse, sys, time, threading, json
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ultralytics import YOLO

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
CONF       = 0.10
IOU        = 0.5
TILE_SIZE  = 640
OVERLAP    = 0.2
PROCESS_EVERY_N_FRAMES = 8
LOW_THRESHOLD  = 10
HIGH_THRESHOLD = 25

# ── GLOBALS ───────────────────────────────────────────────────────────────────
history    = []
stop_event = threading.Event()

state = {
    "running": False,
    "frame_id": 0,
    "total_count": 0,
    "status": "loading",
    "error": None,
}
state_lock = threading.Lock()

# Latest JPEG frames for each stream
frames = {
    "video": None,
    "zone":  None,
    "dot":   None,
    "graph": None,
}
frames_lock = threading.Lock()

# ── DETECTION UTILS ───────────────────────────────────────────────────────────
def nms(boxes, scores):
    if not boxes:
        return []
    boxes_xywh = [[x1, y1, x2-x1, y2-y1] for x1,y1,x2,y2 in boxes]
    idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.0, 0.4)
    if len(idx) == 0:
        return []
    return [boxes[i] for i in idx.flatten()]

def tile_detect(frame, model):
    h, w, _ = frame.shape
    step = int(TILE_SIZE * (1 - OVERLAP))
    boxes, scores = [], []
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = frame[y:y+TILE_SIZE, x:x+TILE_SIZE]
            if tile.shape[0] < 100 or tile.shape[1] < 100:
                continue
            res = model(tile, conf=CONF, iou=IOU, verbose=False)
            for r in res:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    if int(b.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    boxes.append([x1+x, y1+y, x2+x, y2+y])
                    scores.append(float(b.conf[0]))
    return boxes, scores

# ── RENDERING ─────────────────────────────────────────────────────────────────
def split_view(frame, boxes):
    h, w   = frame.shape[:2]
    zh, zw = h//2, w//3
    out    = np.zeros_like(frame)
    zid    = 1
    for r in range(2):
        for c in range(3):
            y1, y2 = r*zh, (r+1)*zh
            x1, x2 = c*zw, (c+1)*zw
            zone   = frame[y1:y2, x1:x2].copy()
            count  = 0
            for (bx1, by1, bx2, by2) in boxes:
                cx, cy = (bx1+bx2)//2, (by1+by2)//2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    count += 1
                    cv2.rectangle(zone, (bx1-x1, by1-y1), (bx2-x1, by2-y1), (0,255,0), 2)
            if count < LOW_THRESHOLD:
                col, lab = (0,255,0), "LOW"
            elif count < HIGH_THRESHOLD:
                col, lab = (0,165,255), "MEDIUM"
            else:
                col, lab = (0,0,255), "HIGH"
            cv2.rectangle(zone, (0,0), (zw,zh), col, 3)
            cv2.putText(zone, f"Z{zid}:{count} ({lab})", (10,30), 0, 0.7, col, 2)
            out[y1:y2, x1:x2] = zone
            zid += 1
    cv2.putText(out, f"TOTAL: {len(boxes)}", (40,80), 0, 2, (0,0,255), 4)
    return out

def zone_twin_img(boxes, shape):
    h, w   = shape[:2]
    zh, zw = h//2, w//3
    twin   = np.zeros((300, 600, 3), dtype=np.uint8)
    for i in range(6):
        r, c   = i//3, i%3
        y1, y2 = r*zh, (r+1)*zh
        x1, x2 = c*zw, (c+1)*zw
        cnt    = 0
        for (bx1, by1, bx2, by2) in boxes:
            cx, cy = (bx1+bx2)//2, (by1+by2)//2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                cnt += 1
        if cnt < LOW_THRESHOLD:
            col = (0,255,0)
        elif cnt < HIGH_THRESHOLD:
            col = (0,165,255)
        else:
            col = (0,0,255)
        cv2.rectangle(twin, (c*200, r*150), (c*200+200, r*150+150), col, -1)
        cv2.rectangle(twin, (c*200, r*150), (c*200+200, r*150+150), (255,255,255), 2)
        cv2.putText(twin, f"Z{i+1}:{cnt}", (c*200+10, r*150+70), 0, 1, (0,0,0), 2)
    return twin

def dot_twin_img(boxes, shape):
    h, w  = shape[:2]
    grid  = np.zeros((20, 20))
    ch, cw = h//20, w//20
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        grid[min(cy//ch, 19)][min(cx//cw, 19)] += 1
    vis = np.zeros((500, 500, 3), dtype=np.uint8)
    for rr in range(20):
        for cc in range(20):
            v = grid[rr][cc]
            if v == 0:
                col = (120,120,120)
            elif v < 2:
                col = (0,255,0)
            elif v < 4:
                col = (0,165,255)
            else:
                col = (0,0,255)
            cv2.rectangle(vis, (cc*25, rr*25), (cc*25+25, rr*25+25), col, -1)
            cv2.rectangle(vis, (cc*25, rr*25), (cc*25+25, rr*25+25), (0,0,0), 1)
    return vis

def draw_graph():
    h, w = 200, 600
    img  = np.zeros((h, w, 3), dtype=np.uint8)
    if len(history) > 1:
        maxv = max(history) + 1
        for i in range(1, len(history)):
            x1 = int((i-1)/len(history)*w)
            x2 = int(i/len(history)*w)
            y1 = h - int(history[i-1]/maxv*h)
            y2 = h - int(history[i]/maxv*h)
            cv2.line(img, (x1,y1), (x2,y2), (0,255,255), 2)
    cv2.putText(img, "CROWD COUNT HISTORY", (10, 20), 0, 0.5, (90,140,170), 1)
    if history:
        cv2.putText(img, f"Current: {history[-1]}", (10, h-10), 0, 0.5, (0,212,255), 1)
    return img

def to_jpg(img, quality=80):
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf)

# ── PROCESSING THREAD ─────────────────────────────────────────────────────────
def process_video(video_path):
    global history

    print(f"[VID-TWIN] Loading YOLO model from {MODEL_PATH} ...", flush=True)
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        with state_lock:
            state["status"] = "error"
            state["error"]  = str(e)
        print(f"[VID-TWIN] Model load error: {e}", flush=True)
        return

    print(f"[VID-TWIN] Opening video: {video_path}", flush=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with state_lock:
            state["status"] = "error"
            state["error"]  = f"Cannot open video: {video_path}"
        print(f"[VID-TWIN] Cannot open video: {video_path}", flush=True)
        return

    with state_lock:
        state["status"]  = "running"
        state["running"] = True

    frame_id   = 0
    prev_boxes = []
    history.clear()

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            ret, frame = cap.read()
            if not ret:
                break

        frame_id += 1
        frame = cv2.resize(frame, (1280, 720))

        if frame_id % PROCESS_EVERY_N_FRAMES == 0:
            b, s = tile_detect(frame, model)
            prev_boxes = nms(b, s)

        boxes = prev_boxes
        history.append(len(boxes))
        if len(history) > 100:
            history.pop(0)

        vid_frame  = split_view(frame, boxes)
        zone_frame = zone_twin_img(boxes, frame.shape)
        dot_frame  = dot_twin_img(boxes, frame.shape)
        graph_frame = draw_graph()

        with frames_lock:
            frames["video"] = to_jpg(vid_frame)
            frames["zone"]  = to_jpg(zone_frame)
            frames["dot"]   = to_jpg(dot_frame)
            frames["graph"] = to_jpg(graph_frame)

        with state_lock:
            state["frame_id"]    = frame_id
            state["total_count"] = len(boxes)

        print(
            f"[VID-TWIN] f{frame_id:05d} | people={len(boxes)}",
            flush=True
        )

    cap.release()
    with state_lock:
        state["status"]  = "stopped"
        state["running"] = False
    print("[VID-TWIN] Processing complete.", flush=True)

# ── HTTP HANDLER ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access log

    def do_GET(self):
        p = self.path.split('?')[0]   # strip query string

        if p == '/status':
            with state_lock:
                data = dict(state)
                data["history_len"] = len(history)
            self._json(data)

        elif p.startswith('/stream/'):
            kind = p.split('/stream/')[1]
            if kind not in frames:
                self._json({"error": "unknown stream"}, 404)
                return
            self._mjpeg_stream(kind)

        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == '/stop':
            stop_event.set()
            with state_lock:
                state["running"] = False
            self._json({"message": "stop requested"})
        else:
            self._json({"error": "not found"}, 404)

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg_stream(self, kind):
        """Stream MJPEG frames for the requested kind."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        placeholder = to_jpg(np.zeros((200, 400, 3), dtype=np.uint8))

        try:
            while not stop_event.is_set():
                with frames_lock:
                    jpg = frames.get(kind) or placeholder

                msg = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    jpg + b'\r\n'
                )
                try:
                    self.wfile.write(msg)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.04)  # ~25 fps
        except Exception:
            pass

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--port',  type=int, default=5004, help='Port to listen on')
    args = parser.parse_args()

    t = threading.Thread(target=process_video, args=(args.video,), daemon=True)
    t.start()

    print(f"[VID-TWIN] HTTP server on port {args.port}", flush=True)
    server = ThreadingHTTPServer(('', args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        stop_event.set()
