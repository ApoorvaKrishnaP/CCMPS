"""
utils/helpers.py
================
Shared utility functions for the crowd risk prediction system.
"""

import logging
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional
import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure production-style logging for the entire system.

    Args:
        log_level: "DEBUG", "INFO", "WARNING", "ERROR".
        log_file: Optional path to write logs to file.

    Returns:
        Root logger.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy library logs
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)

    return logging.getLogger("crowd_risk")


def pixels_to_meters(pixels: float, pixels_per_meter: float = 20.0) -> float:
    """Convert pixel distance to meters using camera calibration."""
    return pixels / pixels_per_meter


def compute_iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes [x1, y1, x2, y2].

    Args:
        bbox_a, bbox_b: Arrays of shape (4,).

    Returns:
        IoU value in [0, 1].
    """
    xx1 = max(bbox_a[0], bbox_b[0])
    yy1 = max(bbox_a[1], bbox_b[1])
    xx2 = min(bbox_a[2], bbox_b[2])
    yy2 = min(bbox_a[3], bbox_b[3])

    inter_w = max(0, xx2 - xx1)
    inter_h = max(0, yy2 - yy1)
    inter = inter_w * inter_h

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter

    return inter / max(union, 1e-6)


def timestamp_to_str(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    return str(datetime.timedelta(seconds=int(seconds)))


def load_config(config_path: str) -> dict:
    """Load JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value to [low, high]."""
    return max(low, min(high, value))


class RunningStats:
    """
    Welford's online algorithm for running mean and variance.
    Used for feature normalization without storing all values.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / max(self.n - 1, 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5
