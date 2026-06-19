import cv2
import json
import time
import math
import numpy as np
from services.api.app.services.math_helpers import dir_entropy, flow_conflict

# Aligning with your training framework's exact feature schema
FEATURE_COLS = [
    "people_count", "density", "avg_speed_mps", "stagnation_ratio",
    "avg_dwell_time_sec", "flow_conflict_ratio", "directional_entropy",
    "acceleration_variance", "zone_transitions", "inflow", "outflow",
]

def stream_crowd_predictions(video_path: str, ml_models: dict, horizon_sec: int = 30):
    """
    Background generator engine. Processes raw video frames, tracks pedestrians,
    calculates 11 spatial metrics, extrapolates trends forward by horizon_sec,
    and yields real-time JSON predictive strings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield f"data: {json.dumps({'error': f'Failed to open video source: {video_path}'})}\n\n"
        return

    # Extract configurations from video properties
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Default matching dimensions from your test script (20m x 12m)
    area_m2 = 20.0 * 12.0
    ppm = ((frame_w / 20.0) + (frame_h / 12.0)) / 2
    count_y = int(frame_h * 0.5)
    frames_per_s = max(1, int(round(src_fps)))

    # Pull pre-loaded model weights from global RAM cache
    yolo_model = ml_models["yolo"]
    gru_model = ml_models["gru"]
    scaler = ml_models["scaler"]
    encoder = ml_models["encoder"]
    classes = list(encoder.classes_)

    # Pipeline tracking metrics state
    prev_pos = {}
    track_side = {}
    in_count = 0
    out_count = 0

    frame_counts = []
    speeds_buf = []
    stagnant_buf = []
    directions_buf = []
    current_sec_ids = set()
    prev_sec_ids = set()

    # 6-second window historical buffer for time-series trend forecasting
    raw_feature_history = []
    
    frame_id = 0
    second_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Process every 5th frame to maintain inference throughput speed
            if frame_id % 5 != 0:
                continue

            # Run detection and tracking inside a single unified YOLO pass
            results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
            
            n_frame = 0
            stag_frame = 0

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids  = results[0].boxes.id.cpu().numpy().astype(int)

                for box, tid in zip(boxes_xyxy, track_ids):
                    l, t_, r, b = box
                    cx, cy = (l + r) // 2, (t_ + b) // 2
                    n_frame += 1
                    current_sec_ids.add(tid)

                    # Compute tracking movement velocities
                    if tid in prev_pos:
                        px, py = prev_pos[tid]
                        dm = math.sqrt((cx - px) ** 2 + (cy - py) ** 2) / ppm
                        spd = dm / (5.0 / src_fps)  # based on PROCESS_EVERY frame step
                        speeds_buf.append(spd)
                        if spd < 0.15:  # STAGNATION_THRESH
                            stag_frame += 1
                        
                        dx, dy = cx - px, cy - py
                        if abs(dx) > 0.5 or abs(dy) > 0.5:
                            directions_buf.append(math.degrees(math.atan2(-dy, dx)) % 360)
                    
                    prev_pos[tid] = (cx, cy)

                    # Line crossing logic (Inflow/Outflow tracking metrics)
                    cur_side = cy > count_y
                    if tid not in track_side:
                        track_side[tid] = cur_side
                    if track_side[tid] != cur_side:
                        if cur_side:
                            in_count += 1
                        else:
                            out_count += 1
                        track_side[tid] = cur_side

            frame_counts.append(n_frame)
            stagnant_buf.append(stag_frame)

            # --- Once Per Second Aggregation Boundary ---
            if frame_id % frames_per_s == 0:
                second_idx += 1
                
                people = float(np.mean(frame_counts)) if frame_counts else 0.0
                speed = float(np.mean(speeds_buf)) if speeds_buf else 0.0
                stag = sum(stagnant_buf) / max(sum(frame_counts), 1)
                
                inflow_s = float(len(current_sec_ids - prev_sec_ids))
                outflow_s = float(len(prev_sec_ids - current_sec_ids))
                prev_sec_ids = set(current_sec_ids)

                # Compile all 11 features exactly mapping to FEATURE_COLS layout
                feat_dict = {
                    "people_count": people,
                    "density": people / area_m2,
                    "avg_speed_mps": speed,
                    "stagnation_ratio": stag,
                    "avg_dwell_time_sec": 20.0,  # Constant baseline fallback
                    "flow_conflict_ratio": flow_conflict(directions_buf),
                    "directional_entropy": dir_entropy(directions_buf),
                    "acceleration_variance": 0.0,
                    "zone_transitions": inflow_s + outflow_s,
                    "inflow": inflow_s,
                    "outflow": outflow_s,
                }

                raw_feature_history.append(feat_dict)
                if len(raw_feature_history) > 6:  # Maintain max SEQUENCE_LENGTH
                    raw_feature_history.pop(0)

                # Clear short-term buffers for next second interval
                frame_counts.clear()
                speeds_buf.clear()
                stagnant_buf.clear()
                directions_buf.clear()
                current_sec_ids.clear()

                # Warmup check: Need 6 full seconds of context data history to forecast trends
                if len(raw_feature_history) < 6:
                    yield f"data: {json.dumps({'status': 'warming up', 'seconds_left': 6 - len(raw_feature_history)})}\n\n"
                    continue

                # --- Execute Linear Extrapolation Trend Analysis (Future Projection) ---
                matrix = np.array([[d[c] for c in FEATURE_COLS] for d in raw_feature_history], dtype=np.float32)
                t_steps = np.arange(6, dtype=np.float32)
                slopes = np.zeros(matrix.shape[1], dtype=np.float32)
                
                for fi in range(matrix.shape[1]):
                    slopes[fi] = np.polyfit(t_steps, matrix[:, fi], 1)[0]

                last_known_metrics = matrix[-1]
                projected_window = np.zeros((6, matrix.shape[1]), dtype=np.float32)
                start_offset = horizon_sec - 6 + 1

                for si in range(6):
                    delta = start_offset + si
                    projected_window[si] = last_known_metrics + slopes * delta

                # Clip values to physically realistic real-world bounds
                projected_window[:, FEATURE_COLS.index("density")] = np.clip(projected_window[:, FEATURE_COLS.index("density")], 0, 6.0)
                projected_window[:, FEATURE_COLS.index("avg_speed_mps")] = np.clip(projected_window[:, FEATURE_COLS.index("avg_speed_mps")], 0, 3.0)
                projected_window[:, FEATURE_COLS.index("stagnation_ratio")] = np.clip(projected_window[:, FEATURE_COLS.index("stagnation_ratio")], 0, 1.0)
                projected_window[:, FEATURE_COLS.index("flow_conflict_ratio")] = np.clip(projected_window[:, FEATURE_COLS.index("flow_conflict_ratio")], 0, 1.0)
                projected_window[:, FEATURE_COLS.index("directional_entropy")] = np.clip(projected_window[:, FEATURE_COLS.index("directional_entropy")], 0, 1.0)
                projected_window[:, FEATURE_COLS.index("people_count")] = np.clip(projected_window[:, FEATURE_COLS.index("people_count")], 0, 300)

                # Scale the projected sequence array through the Standard Scaler
                scaled_sequence = scaler.transform(projected_window)  # Output shape: (6, 11)
                gru_input_tensor = scaled_sequence[np.newaxis]        # Reshape to 3D tensor: (1, 6, 11)

                # --- Run GRU Sequence Prediction Inference ---
                probabilities = gru_model.predict(gru_input_tensor, verbose=0)[0]
                class_idx = int(np.argmax(probabilities))
                raw_label = classes[class_idx]
                confidence = float(probabilities[class_idx])

                # Construct payload streaming contract
                payload = {
                    "second": second_idx,
                    "current_count": round(feat_dict["people_count"], 1),
                    "density": round(feat_dict["density"], 3),
                    "avg_speed": round(feat_dict["avg_speed_mps"], 2),
                    "stagnation": round(feat_dict["stagnation_ratio"], 2),
                    "forecast_horizon": horizon_sec,
                    "predicted_risk_level": raw_label,
                    "confidence_score": round(confidence, 3),
                    "timestamp": time.time()
                }

                yield f"data: {json.dumps(payload)}\n\n"

    finally:
        cap.release()