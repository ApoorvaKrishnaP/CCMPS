import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "crowd_vid.mp4"

CONF = 0.10
IOU = 0.5

TILE_SIZE = 640
OVERLAP = 0.2

USE_ROI = True
ROI = (350, 720, 400, 1000)

# -------- ZONE CONFIG --------
USE_ZONES = True
GRID_ROWS = 2
GRID_COLS = 3

LOW_THRESHOLD = 10
HIGH_THRESHOLD = 25

# -------- REALTIME CONTROL --------
PROCESS_EVERY_N_FRAMES = 8
MAX_SKIP = 12
# ----------------------------------------

model = YOLO(MODEL_PATH)


# -------- NMS --------
def nms_opencv(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return []

    boxes_np = np.array(boxes)
    scores_np = np.array(scores)

    boxes_xywh = []
    for (x1, y1, x2, y2) in boxes_np:
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh,
        scores_np.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold
    )

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])

    return final_boxes


# -------- TILE DETECTION --------
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
                    if int(box.cls[0]) != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])

                    boxes_all.append([x1 + x, y1 + y, x2 + x, y2 + y])
                    scores_all.append(score)

    return boxes_all, scores_all


# -------- ZONE HELPERS --------
def count_in_zone(boxes, zone):
    y1, y2, x1, x2 = zone
    count = 0

    for (bx1, by1, bx2, by2) in boxes:
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        if x1 <= cx <= x2 and y1 <= cy <= y2:
            count += 1

    return count


def get_zones(frame_shape, rows, cols):
    h, w = frame_shape[:2]
    zones = []

    zone_h = h // rows
    zone_w = w // cols

    for r in range(rows):
        for c in range(cols):
            y1 = r * zone_h
            y2 = (r + 1) * zone_h
            x1 = c * zone_w
            x2 = (c + 1) * zone_w
            zones.append((y1, y2, x1, x2))

    return zones


def classify_risk(count):
    if count < LOW_THRESHOLD:
        return "LOW", (0, 255, 0)
    elif count < HIGH_THRESHOLD:
        return "MEDIUM", (0, 165, 255)
    else:
        return "HIGH", (0, 0, 255)


# ---------------- MAIN ----------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Video not opening")
    exit()

prev_boxes = []
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video finished")
        break

    frame_id += 1
    frame = cv2.resize(frame, (1280, 720))

    is_processed = (frame_id % PROCESS_EVERY_N_FRAMES == 0)

    if is_processed:
        boxes, scores = tile_detect(frame)

        if USE_ROI:
            y1, y2, x1, x2 = ROI
            roi = frame[y1:y2, x1:x2]

            roi_boxes, roi_scores = tile_detect(roi)

            for (bx1, by1, bx2, by2) in roi_boxes:
                boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])

            scores.extend(roi_scores)

        final_boxes = nms_opencv(boxes, scores, iou_threshold=0.4)
        prev_boxes = final_boxes
    else:
        final_boxes = prev_boxes

    count = len(final_boxes)

    # ---- ZONE ANALYSIS ----
    if USE_ZONES:
        zones = get_zones(frame.shape, GRID_ROWS, GRID_COLS)
        zone_outputs = []

        for idx, zone in enumerate(zones):
            z_count = count_in_zone(final_boxes, zone)
            risk, color = classify_risk(z_count)
            zone_outputs.append((idx + 1, z_count, risk, zone, color))

        print(f"\nFrame {frame_id} | TOTAL: {count}")

        medium_zones = []
        high_zones = []

        for zid, zc, zr, _, _ in zone_outputs:
            print(f"Zone {zid}: Count={zc}")

            if zr == "MEDIUM":
                medium_zones.append((zid, zc))
            elif zr == "HIGH":
                high_zones.append((zid, zc))

        print("MEDIUM RISK ZONES:",
              ", ".join([f"Z{zid}({zc})" for zid, zc in medium_zones]) if medium_zones else "None")

        print("HIGH RISK ZONES:",
              ", ".join([f"Z{zid}({zc})" for zid, zc in high_zones]) if high_zones else "None")

    # ---- DRAW (only when processed) ----
    if is_processed:
        for (x1, y1, x2, y2) in final_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"TOTAL: {count}", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    if USE_ZONES:
        for zid, zc, zr, zone, color in zone_outputs:
            y1, y2, x1, x2 = zone
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Z{zid}: {zc} ({zr})"
            cv2.putText(frame, label, (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Crowd Detection FAST", frame)

    # 🔥 NO DELAY (max speed)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
