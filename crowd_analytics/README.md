# AI Crowd Risk Prediction System — FOOD_COURT_A

A production-style intelligent surveillance pipeline that detects, tracks, and predicts crowd congestion risk using computer vision and temporal deep learning.

---

## Architecture Overview

```
Surveillance Video
      │
      ▼
YOLOv8 Person Detection          [detector/yolo_detector.py]
      │
      ▼
DeepSORT Persistent Tracking     [tracker/deep_sort_tracker.py]
      │
      ▼  (per-frame)
ZoneAnalytics — 1s Aggregation   [analytics/trajectory_analytics.py]
      │
      ▼  (per second)
Feature Vector (11 features)
      │
      ▼
Temporal Buffer (6 timesteps)    [inference/realtime_predictor.py]
      │
      ▼
GRU Risk Classifier              [models/gru_model.py]
      │
      ▼
SAFE / WARNING / HIGH
      │
      ▼
Real-Time Overlay                [visualization/overlay.py]
```

---

## Monitored Zone

**FOOD_COURT_A** — 150 m², 120-person capacity, full-frame coverage.

---

## Features Extracted Per Second

| Feature | Description |
|---|---|
| `people_count` | Average occupancy over 24 frames |
| `density` | people / zone area (m²) |
| `avg_speed_mps` | Mean pedestrian speed |
| `stagnation_ratio` | Fraction of people nearly stationary |
| `avg_dwell_time_sec` | Mean time-in-zone per track |
| `flow_conflict_ratio` | Fraction of opposing movement pairs |
| `directional_entropy` | Shannon entropy of direction histogram |
| `acceleration_variance` | Variance of scalar accelerations |
| `zone_transitions` | Entries + exits per second |
| `inflow` | New track IDs entering zone |
| `outflow` | Track IDs that left zone |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic training data + train GRU
```bash
python generate_dataset.py --seconds 7200 --train
```

### 3. Run real-time inference on video
```bash
python inference/run_inference.py --source path/to/video.mp4 --output outputs/annotated.mp4
```

### 4. Run on live webcam
```bash
python inference/run_inference.py --source 0
```

### 5. Extract features from real footage (for retraining)
```bash
python extract_features.py --video footage.mp4 --output data/real_features.csv
python train_gru.py --csv data/real_features.csv
```

---

## Project Structure

```
crowd_risk_system/
├── detector/
│   └── yolo_detector.py        YOLOv8 person detection wrapper
├── tracker/
│   └── deep_sort_tracker.py    DeepSORT tracker + trajectory bookkeeping
├── analytics/
│   ├── trajectory_analytics.py  1-second feature aggregation engine
│   └── risk_labeler.py          Rule-based ground-truth labelling
├── feature_extraction/
│   └── pipeline.py              Video → CSV feature extraction pipeline
├── dataset_generation/
│   └── synthetic_generator.py   Synthetic dataset with realistic patterns
├── models/
│   └── gru_model.py             GRU architecture (TensorFlow/Keras)
├── inference/
│   ├── realtime_predictor.py    Temporal buffer + GRU inference
│   └── run_inference.py         Full real-time pipeline CLI
├── visualization/
│   └── overlay.py               Surveillance overlays (boxes, heatmap, panel)
├── utils/
│   └── config.py                Centralised configuration
├── generate_dataset.py          Entry point: generate synthetic data
├── extract_features.py          Entry point: extract from real video
├── train_gru.py                 Entry point: train GRU model
├── data/                        Generated CSVs
├── models/                      Saved model + scaler + encoder
└── outputs/                     Training plots, annotated videos
```

---

## Why GRU?

Crowd congestion is a **temporal process**. It builds across multiple seconds:

1. Density rises slightly, movement slows
2. Stagnation ratio increases
3. Flow conflicts emerge between entering and exiting crowds
4. Full congestion — HIGH risk

A single frame cannot capture this buildup. The GRU receives a **6-second sliding window** of feature vectors and learns the temporal signature of congestion onset. This allows it to raise WARNING before HIGH risk materialises — giving operators time to respond.

---

## Risk Labels

| Label | Trigger |
|---|---|
| SAFE | density < 0.30, stagnation < 0.35 |
| WARNING | density 0.30–0.60, or stagnation 0.35–0.65 |
| HIGH | density > 0.60, or stagnation > 0.65, or high conflict + density |

---

## Outputs

After training:
- `models/gru_crowd_risk.keras` — trained GRU model
- `models/feature_scaler.pkl` — fitted MinMaxScaler
- `models/label_encoder.pkl` — LabelEncoder
- `outputs/learning_curves.png` — loss + accuracy plots
- `outputs/confusion_matrix.png` — validation confusion matrix
- `outputs/prob_distribution.png` — per-class probability distributions
