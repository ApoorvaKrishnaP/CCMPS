# Combined Crowd Analytics System (YOLO + DeepSORT)

import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "crowd_vid.mp4"

CONF = 0.25
IOU = 0.45

TILE_SIZE = 640
OVERLAP = 0.2

REAL_WORLD_WIDTH_METERS = 20
REAL_WORLD_HEIGHT_METERS = 12

PROCESS_EVERY_N_FRAMES = 5

# Line for inflow/outflow counting
COUNT_LINE_Y = 350

# ======================================================
# LOAD MODEL + TRACKER
# ======================================================
model = YOLO(MODEL_PATH)

tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0
)

# ======================================================
# NMS FUNCTION
# ======================================================
def nms_opencv(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return [], []

    boxes_np = np.array(boxes)

    boxes_xywh = []
    for (x1, y1, x2, y2) in boxes_np:
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh,
        scores,
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )

    final_boxes = []
    final_scores = []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])

    return final_boxes, final_scores

# ======================================================
# TILE DETECTION
# ======================================================
def tile_detect(frame):
    h, w, _ = frame.shape

    step = int(TILE_SIZE * (1 - OVERLAP))

    boxes_all = []
    scores_all = []

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
                    cls = int(box.cls[0])

                    if cls != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])

                    boxes_all.append([
                        x1 + x,
                        y1 + y,
                        x2 + x,
                        y2 + y
                    ])

                    scores_all.append(score)

    return boxes_all, scores_all

# ======================================================
# MAIN
# ======================================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error opening video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ======================================================
# REAL WORLD SCALING
# ======================================================
pixels_per_meter_x = frame_width / REAL_WORLD_WIDTH_METERS
pixels_per_meter_y = frame_height / REAL_WORLD_HEIGHT_METERS

avg_pixels_per_meter = (pixels_per_meter_x + pixels_per_meter_y) / 2

area_m2 = REAL_WORLD_WIDTH_METERS * REAL_WORLD_HEIGHT_METERS

# ======================================================
# STORAGE
# ======================================================
prev_positions = {}
track_last_side = {}

in_count = 0
out_count = 0

start_time = time.time()

frame_id = 0

# ======================================================
# LOOP
# ======================================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended")
        break

    frame_id += 1

    # ==========================================
    # PROCESS FRAME
    # ==========================================
    if frame_id % PROCESS_EVERY_N_FRAMES == 0:

        boxes, scores = tile_detect(frame)

        final_boxes, final_scores = nms_opencv(boxes, scores)

        # ==========================================
        # CONVERT FOR DEEPSORT
        # ==========================================
        detections = []

        for (x1, y1, x2, y2), score in zip(final_boxes, final_scores):

            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                score,
                'person'
            ))

        # ==========================================
        # TRACKING
        # ==========================================
        tracks = tracker.update_tracks(detections, frame=frame)

        people_count = 0
        speeds = []

        # ==========================================
        # PROCESS TRACKS
        # ==========================================
        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id

            l, t, r, b = map(int, track.to_ltrb())

            cx = (l + r) // 2
            cy = (t + b) // 2

            people_count += 1

            # ======================================
            # SPEED CALCULATION
            # ======================================
            if track_id in prev_positions:

                prev_x, prev_y = prev_positions[track_id]

                dx_pixels = cx - prev_x
                dy_pixels = cy - prev_y

                pixel_distance = np.sqrt(dx_pixels**2 + dy_pixels**2)

                distance_meters = pixel_distance / avg_pixels_per_meter

                time_seconds = PROCESS_EVERY_N_FRAMES / fps

                speed_mps = distance_meters / time_seconds

                speeds.append(speed_mps)

            prev_positions[track_id] = (cx, cy)

            # ======================================
            # INFLOW / OUTFLOW
            # ======================================
            current_side = cy > COUNT_LINE_Y

            if track_id not in track_last_side:
                track_last_side[track_id] = current_side

            previous_side = track_last_side[track_id]

            if previous_side != current_side:

                if current_side:
                    inflow_direction = True
                else:
                    inflow_direction = False

                if inflow_direction:
                    in_count += 1
                else:
                    out_count += 1

                track_last_side[track_id] = current_side

            # ======================================
            # DRAW TRACKS
            # ======================================
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"ID {track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ==========================================
        # METRICS
        # ==========================================

        # Density = people / m²
        density = people_count / area_m2

        # Average speed = m/s
        avg_speed = np.mean(speeds) if len(speeds) > 0 else 0

        # Elapsed time in minutes
        elapsed_minutes = (time.time() - start_time) / 60

        if elapsed_minutes == 0:
            elapsed_minutes = 1 / 60

        # Inflow = people/min
        inflow = in_count / elapsed_minutes

        # Outflow = people/min
        outflow = out_count / elapsed_minutes

        # ==========================================
        # DISPLAY METRICS
        # ==========================================
        cv2.putText(
            frame,
            f"People: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Density: {density:.3f} p/m2",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Avg Speed: {avg_speed:.2f} m/s",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Inflow: {inflow:.2f} people/min",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Outflow: {outflow:.2f} people/min",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Draw counting line
        cv2.line(
            frame,
            (0, COUNT_LINE_Y),
            (frame_width, COUNT_LINE_Y),
            (0, 0, 255),
            2
        )

        # ==========================================
        # TERMINAL OUTPUT
        # ==========================================
        print("=" * 60)
        print(f"Frame: {frame_id}")
        print(f"Density    = {density:.3f} p/m²")
        print(f"Avg Speed  = {avg_speed:.2f} m/s")
        print(f"Inflow     = {inflow:.2f} people/min")
        print(f"Outflow    = {outflow:.2f} people/min")

    # ==========================================
    # SHOW OUTPUT
    # ==========================================
    cv2.imshow("Combined Crowd Analytics", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# ======================================================
# CLEANUP
# ======================================================
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

