"""
yolo_detector.py — YOLOv8 person detector wrapper.

Design choices:
  • We use YOLOv8n (nano) by default for real-time performance.
    Swap to yolov8m/l for higher accuracy if GPU is available.
  • Only class 0 ("person") detections are returned.
  • Confidence and IoU thresholds are tunable at construction time.
  • Returns raw NumPy arrays to keep this layer framework-agnostic.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

class YOLODetector:
    """
    Wraps YOLOv8 for single-class person detection.

    Returns:
        detections: List of (x1, y1, x2, y2, confidence) tuples.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.conf  = conf_threshold
        self.iou   = iou_threshold
        self.device = device
        print(f"[YOLODetector] Loaded {model_name} | conf={conf_threshold} iou={iou_threshold}")

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Run inference on a single BGR frame.

        Args:
            frame: HxWx3 NumPy array (BGR, as returned by cv2.VideoCapture).

        Returns:
            List of (x1, y1, x2, y2, confidence).
            Empty list if no persons detected.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[0],           # class 0 = person in COCO
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append((float(x1), float(y1), float(x2), float(y2), conf))

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List]:
        """Batch inference for offline processing."""
        return [self.detect(f) for f in frames]
