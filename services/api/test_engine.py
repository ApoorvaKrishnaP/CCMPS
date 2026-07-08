import os
import time
from services.api.app.services.analytics_engine import stream_crowd_predictions

# 1. Dynamically locate the absolute path to your weights folder
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "app", "models")

print("[TEST] Initializing mock weight cache with absolute paths...")
from ultralytics import YOLO
import tensorflow as tf
import pickle

# Inject the smart path configurations here
ml_models = {
    "yolo": YOLO(os.path.join(models_dir, "yolov8n.pt")),
    "gru": tf.keras.models.load_model(os.path.join(models_dir, "gru_best.keras")),
}

with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
    ml_models["scaler"] = pickle.load(f)
with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
    ml_models["encoder"] = pickle.load(f)

# 2. Point to a sample video file on your system
# Place your 'crowd_vid.mp4' right inside the 'services/api/' folder
VIDEO_PATH = os.path.join(base_dir, "Crowd1.mp4")  

print("[TEST] Starting generator stream tracking loop...")
# 3. Consume the first 5 frames yielded by your background generator
generator = stream_crowd_predictions(VIDEO_PATH, ml_models)

for i, data_packet in enumerate(generator):
    print(f"\n[FRAME PACKET {i} RECEIVED]:")
    print(data_packet)
    
    if i >= 5: # Break early so your console doesn't flood
        print("\n[TEST] Success! Generator is producing valid real-time payloads.")
        break