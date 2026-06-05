"""
test_gru.py  —  GRU Crowd Risk Predictor with Future Projection
================================================================
- Predicts FUTURE risk (not current) using trend extrapolation
- Shows only the dominant class + confidence
- Hysteresis:  enter HIGH only after 3 consecutive HIGH seconds
               exit  HIGH only after 5 consecutive SAFE  seconds
- Confidence gating: only act on predictions >= 60%

Place in: CCMPS/crowd_analytics/
Run:      python test_gru.py --video path/to/video.mp4
"""

import os, sys, argparse, math, pickle, time
from collections import deque
import numpy as np
import cv2

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--video",   required=True)
parser.add_argument("--width",   type=float, default=20.0)
parser.add_argument("--height",  type=float, default=12.0)
parser.add_argument("--models",  default="models")
parser.add_argument("--horizon", type=int,   default=30,
                    help="Seconds ahead to forecast (default 30)")
args = parser.parse_args()

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH    = os.path.join(SCRIPT_DIR, "yolov8n.pt")
MODELS_DIR   = os.path.join(SCRIPT_DIR, args.models)
KERAS_PATH   = os.path.join(MODELS_DIR, "gru_best.keras")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ── PIPELINE CONSTANTS ────────────────────────────────────────────────────────
CONF              = 0.25
IOU               = 0.45
TILE_SIZE         = 640
OVERLAP           = 0.2
PROCESS_EVERY     = 5
SEQUENCE_LENGTH   = 6
STAGNATION_THRESH = 0.15
MIN_CONFIDENCE    = 0.60   # ignore predictions below this

# Hysteresis thresholds
HIGH_ENTER_STREAK = 3   # consecutive HIGH raw predictions → confirmed HIGH
SAFE_EXIT_STREAK  = 5   # consecutive SAFE raw predictions → exit HIGH

FEATURE_COLS = [
    "people_count", "density", "avg_speed_mps", "stagnation_ratio",
    "avg_dwell_time_sec", "flow_conflict_ratio", "directional_entropy",
    "acceleration_variance", "zone_transitions", "inflow", "outflow",
]

# ── CSV LOG SETUP ─────────────────────────────────────────────────────────────
import csv as _csv
CSV_OUT    = "risk_log.csv"
_csv_file  = open(CSV_OUT, "w", newline="")
_csv_writer = _csv.writer(_csv_file)
_csv_writer.writerow([
    "second", "people", "density", "speed", "stagnation",
    "raw_label", "confidence", "confirmed_risk"
])

# ── LOAD GRU ──────────────────────────────────────────────────────────────────
print("[INFO] Loading GRU model...")
try:
    import tensorflow as tf
    gru_model = tf.keras.models.load_model(KERAS_PATH)
    with open(SCALER_PATH,  "rb") as f: scaler  = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: encoder = pickle.load(f)
    CLASSES = list(encoder.classes_)   # e.g. ['HIGH', 'SAFE', 'WARNING']
    print(f"[INFO] Model ready  |  classes: {CLASSES}")
except Exception as e:
    print(f"[ERROR] {e}"); sys.exit(1)

# ── LOAD YOLO + TRACKER ───────────────────────────────────────────────────────
print("[INFO] Loading YOLO with Native Tracking...")
from ultralytics import YOLO
yolo = YOLO(YOLO_PATH)
print("[INFO] Ready.\n")

# ── OPEN VIDEO ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"[ERROR] Cannot open: {args.video}"); sys.exit(1)

src_fps      = cap.get(cv2.CAP_PROP_FPS) or 24.0
frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
area_m2      = args.width * args.height
ppm          = ((frame_w / args.width) + (frame_h / args.height)) / 2
count_y      = int(frame_h * 0.5)
frames_per_s = max(1, int(round(src_fps)))

print(f"[INFO] {frame_w}x{frame_h} @ {src_fps:.0f}fps  "
      f"zone={args.width}x{args.height}m  horizon=+{args.horizon}s")
print(f"[INFO] Warmup: {SEQUENCE_LENGTH}s  |  Hysteresis: "
      f"HIGH enter={HIGH_ENTER_STREAK}s  exit={SAFE_EXIT_STREAK}s SAFE")
print(f"{'─'*68}\n")

# ── STATE ─────────────────────────────────────────────────────────────────────
prev_pos        = {}
track_side      = {}
in_count        = 0
out_count       = 0

# per-second accumulators
frame_counts    = []
speeds_buf      = []
stagnant_buf    = []
active_buf      = []
directions_buf  = []
current_sec_ids = set()
prev_sec_ids    = set()

# GRU buffer — stores UNSCALED feature dicts for trend analysis
raw_feature_buf  = deque(maxlen=SEQUENCE_LENGTH)   # dicts
scaled_gru_buf   = deque(maxlen=SEQUENCE_LENGTH)   # scaled vecs for GRU input

# Hysteresis state machine
confirmed_risk   = "WARMING UP"
high_streak      = 0   # consecutive raw HIGH predictions
safe_streak      = 0   # consecutive raw SAFE predictions

frame_id         = 0
second_idx       = 0

# ── TIMING ACCUMULATORS ───────────────────────────────────────────────────────
_t = {
    "frame_read":       0.0,
    "tile_detect":      0.0,
    "nms":              0.0,
    "deepsort":         0.0,
    "feature_flush":    0.0,
    "scaler":           0.0,
    "projection":       0.0,
    "gru_predict":      0.0,
    "hysteresis":       0.0,
}
_calls = {k: 0 for k in _t}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def dir_entropy(dirs):
    if not dirs: return 0.0
    h, _ = np.histogram(dirs, bins=8, range=(0,360))
    p    = (h.astype(float)+1e-9); p /= p.sum()
    return float(min(-np.sum(p*np.log(p)) / math.log(8), 1.0))


def flow_conflict(dirs):
    if len(dirs) < 2: return 0.0
    d = np.array(dirs); c = t = 0
    for i in range(len(d)):
        for j in range(i+1, len(d)):
            diff = min(abs(d[i]-d[j]), 360-abs(d[i]-d[j]))
            if diff > 120: c += 1
            t += 1
    return c / max(t, 1)


def flush_second():
    global prev_sec_ids
    people    = float(np.mean(frame_counts)) if frame_counts else 0.0
    speed     = float(np.mean(speeds_buf))   if speeds_buf   else 0.0
    stag      = sum(stagnant_buf) / max(sum(active_buf), 1)
    inflow_s  = float(len(current_sec_ids - prev_sec_ids))
    outflow_s = float(len(prev_sec_ids - current_sec_ids))
    prev_sec_ids = set(current_sec_ids)
    return {
        "people_count":          people,
        "density":               people / area_m2,
        "avg_speed_mps":         speed,
        "stagnation_ratio":      stag,
        "avg_dwell_time_sec":    20.0,
        "flow_conflict_ratio":   flow_conflict(directions_buf),
        "directional_entropy":   dir_entropy(directions_buf),
        "acceleration_variance": 0.0,
        "zone_transitions":      inflow_s + outflow_s,
        "inflow":                inflow_s,
        "outflow":               outflow_s,
    }


# ── FUTURE PROJECTION ─────────────────────────────────────────────────────────
def project_future_sequence(raw_buf, horizon_sec):
    """
    Given the last SEQUENCE_LENGTH raw feature dicts, compute linear trends
    for each feature and extrapolate forward by horizon_sec seconds.
    Returns a scaled sequence of shape (1, SEQUENCE_LENGTH, n_features)
    representing the PROJECTED future window, ready for GRU input.

    Why linear extrapolation?
        The GRU was trained on patterns within 6-second windows.
        We shift the window forward in time by projecting each feature
        along its current slope. If density is rising 0.02/s over the
        last 6s, we assume it keeps rising at that rate for horizon_sec
        more seconds, then take the 6-second window ending at that future point.

    Limitations:
        This is trend projection, not a trained forecaster. It works well
        for gradual buildup (the most operationally important case) and
        will be less accurate for sudden changes. It is honest about this
        — which is why we show confidence alongside the label.
    """
    n = len(raw_buf)
    if n < 2:
        return None

    # Stack features: shape (n, n_features)
    matrix = np.array([[d[c] for c in FEATURE_COLS] for d in raw_buf],
                       dtype=np.float32)

    # Compute per-feature linear slope using least squares over the window
    t = np.arange(n, dtype=np.float32)
    slopes = np.zeros(matrix.shape[1], dtype=np.float32)
    for fi in range(matrix.shape[1]):
        # slope = cov(t, y) / var(t)
        slopes[fi] = np.polyfit(t, matrix[:, fi], 1)[0]

    # Last known values
    last = matrix[-1]

    # Build projected sequence:
    # We want a 6-step window that ENDS at t + horizon_sec.
    # Steps are: (horizon - seq_len + 1) ... horizon  (in seconds ahead)
    projected = np.zeros((SEQUENCE_LENGTH, matrix.shape[1]), dtype=np.float32)
    start_offset = horizon_sec - SEQUENCE_LENGTH + 1
    for si in range(SEQUENCE_LENGTH):
        delta = start_offset + si   # seconds ahead of "now"
        projected[si] = last + slopes * delta

    # Hard-clip features to physically plausible ranges
    projected[:, FEATURE_COLS.index("density")]           = np.clip(projected[:, FEATURE_COLS.index("density")],           0, 6.0)
    projected[:, FEATURE_COLS.index("avg_speed_mps")]     = np.clip(projected[:, FEATURE_COLS.index("avg_speed_mps")],     0, 3.0)
    projected[:, FEATURE_COLS.index("stagnation_ratio")]  = np.clip(projected[:, FEATURE_COLS.index("stagnation_ratio")],  0, 1.0)
    projected[:, FEATURE_COLS.index("flow_conflict_ratio")] = np.clip(projected[:, FEATURE_COLS.index("flow_conflict_ratio")], 0, 1.0)
    projected[:, FEATURE_COLS.index("directional_entropy")] = np.clip(projected[:, FEATURE_COLS.index("directional_entropy")], 0, 1.0)
    projected[:, FEATURE_COLS.index("people_count")]      = np.clip(projected[:, FEATURE_COLS.index("people_count")],      0, 300)
    projected[:, FEATURE_COLS.index("inflow")]            = np.clip(projected[:, FEATURE_COLS.index("inflow")],            0, 50)
    projected[:, FEATURE_COLS.index("outflow")]           = np.clip(projected[:, FEATURE_COLS.index("outflow")],           0, 50)

    # Scale using the fitted scaler
    scaled = scaler.transform(projected)   # (6, 11)
    return scaled[np.newaxis]              # (1, 6, 11)


def run_gru(seq):
    """Run GRU on a (1, 6, 11) scaled sequence. Returns (raw_label, confidence)."""
    probs      = gru_model.predict(seq, verbose=0)[0]   # (n_classes,)
    class_idx  = int(np.argmax(probs))
    raw_label  = CLASSES[class_idx]
    confidence = float(probs[class_idx])
    return raw_label, confidence


# ── HYSTERESIS STATE MACHINE ──────────────────────────────────────────────────
def update_hysteresis(raw_label, confidence):
    """
    Smooths raw per-second GRU outputs into a stable operational risk level.

    Rules:
      - If confidence < MIN_CONFIDENCE → hold previous confirmed state (uncertain)
      - HIGH is confirmed only after HIGH_ENTER_STREAK consecutive HIGH seconds
      - HIGH is exited only after SAFE_EXIT_STREAK  consecutive SAFE  seconds
      - WARNING has no streak requirement (intermediate, less critical)
    """
    global confirmed_risk, high_streak, safe_streak

    if confidence < MIN_CONFIDENCE:
        # Not confident enough — don't change state, just report uncertain
        return confirmed_risk, "LOW CONF"

    # Update streaks
    if raw_label == "HIGH":
        high_streak += 1
        safe_streak  = 0
    elif raw_label == "SAFE":
        safe_streak += 1
        high_streak  = 0
    else:  # WARNING
        high_streak  = 0
        safe_streak  = 0

    # State transitions
    if confirmed_risk != "HIGH":
        if raw_label == "WARNING":
            confirmed_risk = "WARNING"
        if high_streak >= HIGH_ENTER_STREAK:
            confirmed_risk = "HIGH"
            high_streak    = 0
    else:
        # Currently HIGH — only exit on SAFE streak
        if safe_streak >= SAFE_EXIT_STREAK:
            confirmed_risk = "SAFE"
            safe_streak    = 0

    streak_note = ""
    if raw_label == "HIGH" and confirmed_risk != "HIGH":
        streak_note = f"({high_streak}/{HIGH_ENTER_STREAK} to confirm)"
    elif raw_label == "SAFE" and confirmed_risk == "HIGH":
        streak_note = f"({safe_streak}/{SAFE_EXIT_STREAK} to clear)"

    return confirmed_risk, streak_note


# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────
ICONS  = {"SAFE": "🟢", "WARNING": "🟡", "HIGH": "🔴", "WARMING UP": "⏳"}
ARROWS = {"SAFE": "↓", "WARNING": "→", "HIGH": "↑"}

def fmt_line(t, feat, raw_label, conf, confirmed, note):
    icon     = ICONS.get(confirmed, "?")
    raw_icon = ICONS.get(raw_label, "?")
    conf_str = f"{conf*100:.0f}%" if conf is not None else "—"

    line = (f"[t={t:3d}s]  "
            f"people={feat['people_count']:5.1f}  "
            f"density={feat['density']:.3f}  "
            f"speed={feat['avg_speed_mps']:.2f}m/s  "
            f"stag={feat['stagnation_ratio']:.2f}  ")

    if raw_label is None:
        line += f"→  ⏳ warming up ({SEQUENCE_LENGTH - len(raw_feature_buf)}s left)"
    else:
        line += (f"→  raw={raw_icon}{raw_label:<8} conf={conf_str:<5}  "
                 f"CONFIRMED={icon}{confirmed:<8}")
        if note:
            line += f"  {note}"

    return line


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
print(f"{'─'*68}")
print(f"  {'TIME':<8} {'PEOPLE':>7} {'DENSITY':>8} {'SPEED':>8} {'STAG':>6}  "
      f"  RAW PREDICTION    CONFIRMED RISK")
print(f"{'─'*68}")

while True:
    _t0 = time.perf_counter()
    ret, frame = cap.read()
    _t["frame_read"] += time.perf_counter() - _t0
    _calls["frame_read"] += 1
    if not ret:
        break

    frame_id += 1

    if frame_id % PROCESS_EVERY != 0:
        continue

    # ── Detect + Track ────────────────────────────────────────────────────────
    _t0 = time.perf_counter()
    
    # Run object detection and BoTSORT tracking in a single native line
    results = yolo.track(frame, tracker="botsort.yaml", persist=True, verbose=False)
    
    # Update our tracking execution metric timers
    if "botsort" not in _t: _t["botsort"] = 0.0; _calls["botsort"] = 0
    _t["botsort"] += time.perf_counter() - _t0
    _calls["botsort"] += 1

    n_frame = 0
    stag_frame = 0

    # Ensure detections and track IDs actually exist in this frame
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids  = results[0].boxes.id.cpu().numpy().astype(int)
        class_ids  = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, tid, cls_id in zip(boxes_xyxy, track_ids, class_ids):
            if cls_id != 0: continue  # Only track 'person' class (class index 0)
            
            l, t_, r, b = box
            cx, cy      = (l + r) // 2, (t_ + b) // 2
            n_frame    += 1
            current_sec_ids.add(tid)

            if tid in prev_pos:
                px, py = prev_pos[tid]
                dm     = math.sqrt((cx-px)**2+(cy-py)**2) / ppm
                spd    = dm / (PROCESS_EVERY / src_fps)
                speeds_buf.append(spd)
                if spd < STAGNATION_THRESH: stag_frame += 1
                dx, dy = cx-px, cy-py
                if abs(dx)>0.5 or abs(dy)>0.5:
                    directions_buf.append(math.degrees(math.atan2(-dy,dx))%360)
            prev_pos[tid] = (cx, cy)

            cur_side = cy > count_y
            if tid not in track_side: track_side[tid] = cur_side
            if track_side[tid] != cur_side:
                if cur_side: in_count  += 1
                else:        out_count += 1
                track_side[tid] = cur_side

    frame_counts.append(n_frame)
    stagnant_buf.append(stag_frame)
    active_buf.append(n_frame)

    # ── Once per second ───────────────────────────────────────────────────────
    if frame_id % frames_per_s == 0:
        _t0 = time.perf_counter()
        feat = flush_second()
        _t["feature_flush"] += time.perf_counter() - _t0
        _calls["feature_flush"] += 1
        raw_feature_buf.append(feat)

        # Scale and add to GRU buffer for current-moment reference
        _t0 = time.perf_counter()
        vec_s = scaler.transform(
            np.array([feat[c] for c in FEATURE_COLS], dtype=np.float32).reshape(1,-1)
        )[0]
        _t["scaler"] += time.perf_counter() - _t0
        _calls["scaler"] += 1
        scaled_gru_buf.append(vec_s)

        # Clear per-second accumulators
        frame_counts.clear(); speeds_buf.clear(); stagnant_buf.clear()
        active_buf.clear();   directions_buf.clear(); current_sec_ids.clear()

        second_idx += 1

        # ── Need full buffer before predicting ────────────────────────────────
        if len(raw_feature_buf) < SEQUENCE_LENGTH:
            print(f"[t={second_idx:3d}s]  ⏳ warming up  "
                  f"({SEQUENCE_LENGTH - len(raw_feature_buf)}s until first prediction)")
            continue

        # ── Project features forward by horizon_sec ───────────────────────────
        _t0 = time.perf_counter()
        proj_seq = project_future_sequence(raw_feature_buf, args.horizon)
        _t["projection"] += time.perf_counter() - _t0
        _calls["projection"] += 1
        if proj_seq is None:
            continue

        # ── Run GRU on projected future sequence ──────────────────────────────
        _t0 = time.perf_counter()
        raw_label, confidence = run_gru(proj_seq)
        _t["gru_predict"] += time.perf_counter() - _t0
        _calls["gru_predict"] += 1

        # ── Apply hysteresis ──────────────────────────────────────────────────
        _t0 = time.perf_counter()
        confirmed, note = update_hysteresis(raw_label, confidence)
        _t["hysteresis"] += time.perf_counter() - _t0
        _calls["hysteresis"] += 1

        # ── Save to CSV ───────────────────────────────────────────────────────
        _csv_writer.writerow([
            second_idx,
            round(feat["people_count"],    1),
            round(feat["density"],         3),
            round(feat["avg_speed_mps"],   2),
            round(feat["stagnation_ratio"],2),
            raw_label,
            round(confidence, 3),
            confirmed,
        ])

        # ── Print ─────────────────────────────────────────────────────────────
        icon_raw  = ICONS.get(raw_label, "?")
        icon_conf = ICONS.get(confirmed, "?")
        conf_str  = f"{confidence*100:.0f}%"

        print(f"[t={second_idx:3d}s]  "
              f"people={feat['people_count']:5.1f}  "
              f"density={feat['density']:.3f}  "
              f"speed={feat['avg_speed_mps']:.2f}m/s  "
              f"stag={feat['stagnation_ratio']:.2f}  "
              f"→  {icon_raw}{raw_label:<8} {conf_str:<5}  "
              f"FORECAST +{args.horizon}s: {icon_conf}{confirmed:<8}"
              + (f"  {note}" if note else ""))

        # ── Alert line for confirmed HIGH ─────────────────────────────────────
        if confirmed == "HIGH" and not note:
            print(f"          ⚠️  CONGESTION ALERT — HIGH risk forecast "
                  f"in ~{args.horizon} seconds")

cap.release()
_csv_file.close()
print(f"[INFO] Predictions saved → {CSV_OUT}")
print(f"\n{'─'*68}")
print(f"[DONE] {second_idx} seconds processed.")

# ── TIMING SUMMARY ────────────────────────────────────────────────────────────
total = sum(_t.values())
print(f"\n{'═'*68}")
print(f"  STAGE TIMING SUMMARY  (total wall time in pipeline: {total:.2f}s)")
print(f"{'─'*68}")
print(f"  {'Stage':<20} {'Total(s)':>10} {'Calls':>8} {'Avg(ms)':>10} {'%Total':>8}")
print(f"{'─'*68}")
for stage, elapsed in sorted(_t.items(), key=lambda x: x[1], reverse=True):
    calls   = _calls[stage]
    avg_ms  = (elapsed / calls * 1000) if calls else 0.0
    pct     = (elapsed / total * 100)  if total  else 0.0
    print(f"  {stage:<20} {elapsed:>10.3f} {calls:>8} {avg_ms:>10.2f} {pct:>7.1f}%")
print(f"{'═'*68}")

# ── AUTO-LAUNCH PLOT ──────────────────────────────────────────────────────────
import subprocess as _sp
_plot_script = os.path.join(SCRIPT_DIR, "plot_risk.py")
if os.path.exists(_plot_script):
    print(f"[INFO] Launching graph window...")
    _sp.Popen([sys.executable, _plot_script, "--csv", CSV_OUT])
else:
    print(f"[INFO] Run:  python plot_risk.py --csv {CSV_OUT}")