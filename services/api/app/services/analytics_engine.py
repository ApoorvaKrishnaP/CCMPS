import cv2
import json
import time
import numpy as np
from app.services.math_helpers import dir_entropy, flow_conflict

def stream_crowd_predictions(video_path: str, ml_models: dict):
    """
    Background generator engine. Opens a video pipe, processes frames 
    through YOLOv8 and the GRU model, and yields real-time analytics JSON strings.
    """
    # 1. Open the video file stream via OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield f"data: {json.dumps({'error': 'Failed to open video source'})}\n\n"
        return

    print(f"[ENGINE] Commencing background stream analysis for: {video_path}")
    
    # Extract structural models directly out of our pre-loaded global memory cache
    yolo_model = ml_models["yolo"]
    gru_model = ml_models["gru"]
    scaler = ml_models["scaler"]
    encoder = ml_models["encoder"]

    frame_counter = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Video file ended smoothly
            
            frame_counter += 1
            # To optimize CPU/GPU processing, sample every 2nd or 3rd frame 
            if frame_counter % 2 != 0:
                continue

            # 2. Run Object Detection (YOLOv8)
            # We track classes [0] (person) natively using YOLO's built-in BoT-SORT tracker
            results = yolo_model.track(frame, persist=True, classes=[0], verbose=False)
            
            pedestrian_count = 0
            angles_list = []

            if results[0].boxes and results[0].boxes.id is not None:
                pedestrian_count = len(results[0].boxes.id)
                
                # (Optional structural step): Parse bounding boxes to calculate tracking velocity vectors
                # For this baseline, we will simulate or extract raw heading trajectories 
                # e.g., mapping box movement paths to direct degrees (0-360)
                # Here we populate mock/calculated tracking angles for our mathematical modules
                angles_list = [np.random.uniform(0, 360) for _ in range(pedestrian_count)]

            # 3. Compute Real-Time Spatial Spatial Mathematics
            entropy_value = dir_entropy(angles_list)
            conflict_value = flow_conflict(angles_list)

            # 4. Construct Input Vector for the GRU Predictive Time-Series Model
            # Shape matches your trained framework layout: [pedestrian_count, entropy, conflict]
            raw_features = np.array([[pedestrian_count, entropy_value, conflict_value]])
            scaled_features = scaler.transform(raw_features)
            
            # Reshape input vector to match GRU's expected 3D sequence footprint: [samples, timesteps, features]
            gru_input = np.reshape(scaled_features, (scaled_features.shape[0], 1, scaled_features.shape[1]))

            # 5. Execute Inference Prediction
            gru_prediction = gru_model.predict(gru_input, verbose=0)
            predicted_class_idx = np.argmax(gru_prediction, axis=1)
            risk_label = encoder.inverse_transform(predicted_class_idx)[0]

            # 6. Construct the Network Payload
            payload = {
                "frame": frame_counter,
                "current_count": pedestrian_count,
                "directional_entropy": round(entropy_value, 3),
                "flow_conflict_index": round(conflict_value, 3),
                "predicted_risk_level": risk_label,
                "timestamp": time.time()
            }

            # Server-Sent Events (SSE) standard protocol requires formatting text data prefixes 
            # terminated by two distinct newline breaks (\n\n)
            yield f"data: {json.dumps(payload)}\n\n"
            
            # Tiny sleep constraint to prevent overloading downstream web sockets
            time.sleep(0.03)

    finally:
        # Guarantee hardware resources are cleanly released even on early disconnection drops
        cap.release()
        print(f"[ENGINE] Safely closed video capture channels for: {video_path}")