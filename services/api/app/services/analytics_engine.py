import cv2
import json
import time
import math
import asyncio
import threading
import numpy as np
from ultralytics import YOLO as _YOLO
from services.api.app.services.math_helpers import dir_entropy, flow_conflict

# torch.load (used inside YOLO init) is NOT thread-safe for concurrent calls.
# This lock serializes YOLO loads so zones don't corrupt each other's load.
_yolo_load_lock = threading.Lock()

FEATURE_COLS = [
    "people_count", "density", "avg_speed_mps", "stagnation_ratio",
    "avg_dwell_time_sec", "flow_conflict_ratio", "directional_entropy",
    "acceleration_variance", "zone_transitions", "inflow", "outflow",
]

async def stream_crowd_predictions(video_path: str, ml_models: dict, horizon_sec: int = 30):
    """
    Async generator — each zone runs in its own asyncio.to_thread call so
    multiple zones can process frames concurrently without blocking each other.
    """
    is_rtsp = video_path.lower().startswith("rtsp://")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield f"data: {json.dumps({'error': f'Failed to open source: {video_path}'})}\n\n"
        return

    cap_ref = [cap]   # mutable so _process_one_second can swap in a reconnected cap

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = raw_fps if raw_fps and 0 < raw_fps <= 120 else 25.0
    frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    area_m2   = 20.0 * 12.0
    ppm       = ((frame_w / 20.0) + (frame_h / 12.0)) / 2
    count_y   = int(frame_h * 0.5)
    frames_per_s = max(1, int(round(src_fps)))

    # Load a fresh YOLO per zone (isolated ByteTracker state).
    # _yolo_load_lock serializes concurrent torch.load calls — each zone still
    # gets its own independent model after the load, then processes in parallel.
    yield f"data: {json.dumps({'status': 'starting'})}\n\n"

    def _load_yolo():
        with _yolo_load_lock:
            return _YOLO(ml_models["yolo_path"])
    yolo_model = await asyncio.to_thread(_load_yolo)
    gru_model  = ml_models["gru"]
    scaler     = ml_models["scaler"]
    encoder    = ml_models["encoder"]
    classes    = list(encoder.classes_)

    # All mutable state lives in this dict so the inner closure can reach it
    s = {
        "prev_pos": {}, "track_side": {},
        "frame_counts": [], "speeds_buf": [], "stagnant_buf": [], "directions_buf": [],
        "current_sec_ids": set(), "prev_sec_ids": set(),
        "raw_feature_history": [],
        "positions": [],
        "frame_id": 0, "second_idx": 0,
    }

    def _process_one_second():
        """
        Runs in a thread (via asyncio.to_thread).
        Reads frames until one full video-second has been aggregated,
        then returns the payload dict (or None if the video ended).
        For RTSP sources a failed read triggers one reconnect attempt.
        """
        while True:
            ret, frame = cap_ref[0].read()
            if not ret:
                if is_rtsp:
                    cap_ref[0].release()
                    time.sleep(2)
                    cap_ref[0] = cv2.VideoCapture(video_path)
                    if not cap_ref[0].isOpened():
                        return None          # camera truly unreachable
                    return {"_reconnecting": True}
                return None                  # video file ended

            s["frame_id"] += 1
            if s["frame_id"] % 5 != 0:
                continue

            results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
            s["positions"] = []

            n_frame = 0
            stag_frame = 0

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids  = results[0].boxes.id.cpu().numpy().astype(int)
                frame_positions = []

                for box, tid in zip(boxes_xyxy, track_ids):
                    l, t_, r, b = box
                    cx, cy = (l + r) // 2, (t_ + b) // 2
                    n_frame += 1
                    s["current_sec_ids"].add(tid)
                    frame_positions.append({
                        "id": int(tid),
                        "x": round(cx / max(frame_w, 1), 4),
                        "y": round(cy / max(frame_h, 1), 4),
                    })

                    if tid in s["prev_pos"]:
                        px, py = s["prev_pos"][tid]
                        dm  = math.sqrt((cx - px) ** 2 + (cy - py) ** 2) / ppm
                        spd = dm / (5.0 / src_fps)
                        s["speeds_buf"].append(spd)
                        if spd < 0.15:
                            stag_frame += 1
                        dx, dy = cx - px, cy - py
                        if abs(dx) > 0.5 or abs(dy) > 0.5:
                            s["directions_buf"].append(
                                math.degrees(math.atan2(-dy, dx)) % 360
                            )

                    s["prev_pos"][tid] = (cx, cy)

                    cur_side = cy > count_y
                    if tid not in s["track_side"]:
                        s["track_side"][tid] = cur_side
                    if s["track_side"][tid] != cur_side:
                        s["track_side"][tid] = cur_side

                s["positions"] = frame_positions

            s["frame_counts"].append(n_frame)
            s["stagnant_buf"].append(stag_frame)

            if s["frame_id"] % frames_per_s != 0:
                continue  # haven't finished a full video-second yet

            # ── One-second aggregation boundary ─────────────────────────────
            s["second_idx"] += 1

            people = float(np.mean(s["frame_counts"])) if s["frame_counts"] else 0.0
            speed  = float(np.mean(s["speeds_buf"]))   if s["speeds_buf"]   else 0.0
            stag   = sum(s["stagnant_buf"]) / max(sum(s["frame_counts"]), 1)

            inflow_s  = float(len(s["current_sec_ids"] - s["prev_sec_ids"]))
            outflow_s = float(len(s["prev_sec_ids"] - s["current_sec_ids"]))
            s["prev_sec_ids"] = set(s["current_sec_ids"])

            feat_dict = {
                "people_count":         people,
                "density":              people / area_m2,
                "avg_speed_mps":        speed,
                "stagnation_ratio":     stag,
                "avg_dwell_time_sec":   20.0,
                "flow_conflict_ratio":  flow_conflict(s["directions_buf"]),
                "directional_entropy":  dir_entropy(s["directions_buf"]),
                "acceleration_variance": 0.0,
                "zone_transitions":     inflow_s + outflow_s,
                "inflow":               inflow_s,
                "outflow":              outflow_s,
            }

            s["raw_feature_history"].append(feat_dict)
            if len(s["raw_feature_history"]) > 6:
                s["raw_feature_history"].pop(0)

            s["frame_counts"].clear()
            s["speeds_buf"].clear()
            s["stagnant_buf"].clear()
            s["directions_buf"].clear()
            s["current_sec_ids"].clear()

            if len(s["raw_feature_history"]) < 6:
                # Still warming up — return a warmup message and come back next second
                return {
                    "_warmup": True,
                    "seconds_left": 6 - len(s["raw_feature_history"]),
                    "second": s["second_idx"],
                    "current_count": round(feat_dict["people_count"], 1),
                    "density": round(feat_dict["density"], 3),
                    "avg_speed": round(feat_dict["avg_speed_mps"], 2),
                    "stagnation": round(feat_dict["stagnation_ratio"], 2),
                    "forecast_horizon": horizon_sec,
                    "confidence_score": round(len(s["raw_feature_history"]) / 6.0, 3),
                    "positions": s["positions"],
                    "timestamp": time.time(),
                }

            # ── GRU prediction ───────────────────────────────────────────────
            matrix   = np.array([[d[c] for c in FEATURE_COLS] for d in s["raw_feature_history"]], dtype=np.float32)
            t_steps  = np.arange(6, dtype=np.float32)
            slopes   = np.array([np.polyfit(t_steps, matrix[:, fi], 1)[0] for fi in range(matrix.shape[1])], dtype=np.float32)

            last   = matrix[-1]
            offset = horizon_sec - 6 + 1
            projected = np.array([last + slopes * (offset + si) for si in range(6)], dtype=np.float32)

            projected[:, FEATURE_COLS.index("density")]            = np.clip(projected[:, FEATURE_COLS.index("density")], 0, 6.0)
            projected[:, FEATURE_COLS.index("avg_speed_mps")]      = np.clip(projected[:, FEATURE_COLS.index("avg_speed_mps")], 0, 3.0)
            projected[:, FEATURE_COLS.index("stagnation_ratio")]   = np.clip(projected[:, FEATURE_COLS.index("stagnation_ratio")], 0, 1.0)
            projected[:, FEATURE_COLS.index("flow_conflict_ratio")]= np.clip(projected[:, FEATURE_COLS.index("flow_conflict_ratio")], 0, 1.0)
            projected[:, FEATURE_COLS.index("directional_entropy")]= np.clip(projected[:, FEATURE_COLS.index("directional_entropy")], 0, 1.0)
            projected[:, FEATURE_COLS.index("people_count")]       = np.clip(projected[:, FEATURE_COLS.index("people_count")], 0, 300)

            scaled    = scaler.transform(projected)
            gru_input = scaled[np.newaxis]
            probs     = gru_model.predict(gru_input, verbose=0)[0]
            class_idx = int(np.argmax(probs))

            return {
                "second":              s["second_idx"],
                "current_count":       round(feat_dict["people_count"], 1),
                "density":             round(feat_dict["density"], 3),
                "avg_speed":           round(feat_dict["avg_speed_mps"], 2),
                "stagnation":          round(feat_dict["stagnation_ratio"], 2),
                "forecast_horizon":    horizon_sec,
                "predicted_risk_level": classes[class_idx],
                "confidence_score":    round(float(probs[class_idx]), 3),
                "positions":           s["positions"],
                "timestamp":           time.time(),
            }

    try:
        while True:
            # Each second's worth of frame processing runs in a thread pool thread.
            # asyncio.to_thread releases the event loop between calls so other
            # zones can run their own thread concurrently — true parallel processing.
            result = await asyncio.to_thread(_process_one_second)

            if result is None:
                break  # video ended or RTSP permanently unreachable

            if result.get("_reconnecting"):
                yield f"data: {json.dumps({'status': 'reconnecting'})}\n\n"
                continue

            if result.get("_warmup"):
                result["status"] = "warming_up"
                yield f"data: {json.dumps(result)}\n\n"
                continue

            yield f"data: {json.dumps(result)}\n\n"

    finally:
        cap_ref[0].release()
