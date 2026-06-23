"""
config.py — Central configuration for the Crowd Risk Prediction System.

All zone definitions, feature parameters, model hyperparameters, and
visualization settings are managed here to avoid hardcoded magic values
scattered throughout the codebase.
"""

# ─── Video / Camera ────────────────────────────────────────────────────────────
VIDEO_FPS = 24
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ─── Zone Definition ───────────────────────────────────────────────────────────
# FOOD_COURT_A is the single monitored zone in this deployment.
# Coordinates are (x_min, y_min, x_max, y_max) in pixel space for a 1280×720 feed.
ZONES = {
    "FOOD_COURT_A": {
        "bbox": (50, 50, 1230, 670),   # near-full-frame zone
        "area_m2": 150.0,              # estimated real-world floor area (metres²)
        "max_capacity": 120,           # operational hard cap
        "warning_threshold": 0.60,     # fraction of capacity → WARNING
        "high_threshold": 0.80,        # fraction of capacity → HIGH
    }
}
ACTIVE_ZONE = "FOOD_COURT_A"

# ─── Feature Extraction ─────────────────────────────────────────────────────────
AGGREGATION_WINDOW_SEC = 1      # seconds per feature vector
TRAJECTORY_HISTORY_SEC = 5      # how long we keep each track's trajectory
MIN_TRACK_FRAMES       = 3      # discard tracks shorter than this (noise)
STAGNATION_SPEED_THRESH_MPS = 0.15   # below this → person is stationary
PIXEL_TO_METRE = 0.05                # rough calibration: 1 pixel ≈ 0.05 m

# ─── GRU Model ──────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 6          # timesteps fed into GRU
FEATURE_COLUMNS = [
    "people_count",
    "density",
    "avg_speed_mps",
    "stagnation_ratio",
    "avg_dwell_time_sec",
    "flow_conflict_ratio",
    "directional_entropy",
    "acceleration_variance",
    "zone_transitions",
    "inflow",
    "outflow",
]
LABEL_COLUMN  = "risk_label"
RISK_CLASSES  = ["SAFE", "WARNING", "HIGH"]

GRU_UNITS          = 64
GRU_LAYERS         = 2
DROPOUT_RATE       = 0.3
LEARNING_RATE      = 1e-3
BATCH_SIZE         = 32
MAX_EPOCHS         = 80
EARLY_STOP_PATIENCE= 10

# ─── Inference / Buffer ─────────────────────────────────────────────────────────
PREDICTION_SMOOTHING_WINDOW = 3   # rolling majority-vote over last N predictions

# ─── Paths ──────────────────────────────────────────────────────────────────────
import os, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR        = ROOT / "data"
OUTPUT_DIR      = ROOT / "outputs"
MODEL_DIR       = ROOT / "models"
DATASET_CSV     = DATA_DIR / "crowd_features.csv"
TRAINED_MODEL   = MODEL_DIR / "gru_crowd_risk.keras"
SCALER_PATH     = MODEL_DIR / "feature_scaler.pkl"
ENCODER_PATH    = MODEL_DIR / "label_encoder.pkl"

for _p in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
