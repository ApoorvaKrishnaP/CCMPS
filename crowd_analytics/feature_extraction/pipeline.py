"""
pipeline.py — End-to-end feature extraction pipeline.

Orchestrates: Video → YOLO → DeepSORT → ZoneAnalytics → CSV row.

IMPORTANT AGGREGATION RULE (re-stated for clarity):
  Frame-level detection data is accumulated across all 24 frames of each
  1-second window. A single feature vector is then computed per window.

  This is the ONLY correct way to feed the GRU.
  Frame-level inputs would cause temporal aliasing and prevent the model from
  learning congestion buildup patterns.
"""

from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detector.yolo_detector import YOLODetector
from tracker import get_tracker
from analytics.trajectory_analytics import ZoneAnalytics
from analytics.risk_labeler import label_risk
from utils.config import VIDEO_FPS, FEATURE_COLUMNS, LABEL_COLUMN, DATASET_CSV


class FeatureExtractionPipeline:
    """
    Full pipeline that processes a video file and writes crowd feature CSVs.

    Usage:
        pipeline = FeatureExtractionPipeline()
        pipeline.run(video_path="mall_footage.mp4", output_csv="data/features.csv")
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        conf: float = 0.40,
        device: str = "cpu",
    ) -> None:
        self.detector  = YOLODetector(yolo_model, conf_threshold=conf, device=device)
        # Default tracker: use factory. Set USE_DEEPSORT env var to opt-in.
        self.tracker   = get_tracker()
        self.analytics = ZoneAnalytics()

    def run(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        max_frames: Optional[int] = None,
        label: bool = True,
    ) -> list:
        """
        Process video file, extract features, write CSV.

        Args:
            video_path:  Path to input video.
            output_csv:  Destination CSV. Defaults to DATASET_CSV from config.
            max_frames:  Cap frame count (useful for quick testing).
            label:       If True, add rule-based risk labels to CSV.

        Returns:
            List of feature dicts (one per second of video).
        """
        out_path = Path(output_csv) if output_csv else DATASET_CSV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps_src = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Pipeline] Video: {video_path}  FPS={fps_src:.1f}  Frames={total_frames}")

        fieldnames = ["timestamp"] + FEATURE_COLUMNS + ([LABEL_COLUMN] if label else [])
        records = []
        frame_idx = 0
        second_idx = 0

        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_idx >= max_frames:
                    break

                # ── Per-frame processing ──────────────────────────────────────
                detections = self.detector.detect(frame)
                tracks     = self.tracker.update(detections, frame, frame_idx)
                states     = self.tracker.get_all_states()
                self.analytics.ingest_frame(tracks, states, frame_idx)

                # ── Window flush: once per second ─────────────────────────────
                if frame_idx > 0 and frame_idx % VIDEO_FPS == 0:
                    feat = self.analytics.flush_window()
                    feat["timestamp"] = second_idx
                    if label:
                        feat[LABEL_COLUMN] = label_risk(feat)

                    writer.writerow(feat)
                    fh.flush()
                    records.append(feat)
                    second_idx += 1

                    if second_idx % 60 == 0:
                        print(f"[Pipeline] t={second_idx}s  "
                              f"people={feat['people_count']:.0f}  "
                              f"density={feat['density']:.3f}  "
                              f"risk={feat.get(LABEL_COLUMN,'?')}")

                frame_idx += 1

        cap.release()
        print(f"[Pipeline] Done — {second_idx} feature rows → {out_path}")
        return records
