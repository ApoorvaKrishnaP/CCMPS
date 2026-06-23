"""
inference/realtime_inference.py
================================
Real-time crowd risk inference engine for FOOD_COURT_A.

Pipeline:
    Live Video Frame
    → YOLOv8 Detection
    → CrowdTracker Update
    → CrowdAnalyticsEngine.push_frame()
    → Temporal Buffer
    → GRU Prediction (when buffer full)
    → Risk State Update
    → Visualization

Temporal Buffer Design:
───────────────────────
The system maintains a rolling feature buffer matching the GRU's
sequence_length. When a new 1-second feature vector is computed,
it is appended to the buffer. Once the buffer has enough entries,
the GRU receives a complete temporal sequence for prediction.

This is mandatory for correct GRU operation — feeding a single
vector to a sequence model produces meaningless predictions.

Prediction Smoothing:
─────────────────────
Raw GRU output can oscillate between SAFE/WARNING frame-to-frame.
We apply exponential moving average (EMA) on class probabilities
to stabilize the displayed risk state. This prevents "flickering"
alerts that confuse security operators.
"""

import numpy as np
import json
import pickle
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Risk class colors for visualization (BGR format for OpenCV)
RISK_COLORS = {
    "SAFE":    (0, 200, 60),     # Green
    "WARNING": (0, 165, 255),    # Orange
    "HIGH":    (0, 0, 220),      # Red
    "UNKNOWN": (150, 150, 150),  # Gray
}

RISK_TEXT_COLORS = {
    "SAFE":    (255, 255, 255),
    "WARNING": (255, 255, 255),
    "HIGH":    (255, 255, 255),
    "UNKNOWN": (200, 200, 200),
}


@dataclass
class RiskPrediction:
    """Structured GRU prediction output."""
    risk_class: str               # "SAFE", "WARNING", "HIGH"
    probabilities: Dict[str, float]  # Per-class probabilities
    confidence: float             # Max probability
    timestamp: float
    smoothed: bool = True


class GRURiskPredictor:
    """
    Loads a trained GRU model and provides real-time inference
    with temporal buffering and prediction smoothing.

    Usage:
        predictor = GRURiskPredictor("outputs/models")
        predictor.load()

        # Each second, call with the new feature vector:
        prediction = predictor.predict(feature_vector)
        print(prediction.risk_class)
    """

    def __init__(
        self,
        model_dir: str = "outputs/models",
        ema_alpha: float = 0.35,  # EMA smoothing factor (lower = more smoothing)
    ):
        """
        Args:
            model_dir: Directory containing trained model artifacts.
            ema_alpha: EMA alpha for probability smoothing (0.2-0.5 typical).
        """
        self.model_dir = Path(model_dir)
        self.ema_alpha = ema_alpha

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.config = {}

        # Temporal feature buffer (rolling window)
        self._feature_buffer: deque = deque()
        self._sequence_length = 6

        # Smoothed probabilities (EMA)
        self._smoothed_probs = np.array([1.0, 0.0, 0.0])  # Start as SAFE

        # History for alerting
        self._prediction_history: deque = deque(maxlen=20)

        self._is_loaded = False

    def load(self) -> bool:
        """
        Load model, scaler, and label encoder from disk.

        Returns:
            True if loading succeeded.
        """
        try:
            import tensorflow as tf

            # Load Keras model
            model_path = self.model_dir / "gru_best.keras"
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False

            self.model = tf.keras.models.load_model(str(model_path))
            logger.info("GRU model loaded successfully.")

            # Load scaler
            with open(self.model_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            # Load label encoder
            with open(self.model_dir / "label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

            # Load config
            with open(self.model_dir / "model_config.json", "r") as f:
                self.config = json.load(f)

            self._sequence_length = self.config.get("sequence_length", 6)
            self._feature_buffer = deque(maxlen=self._sequence_length)

            self._is_loaded = True
            logger.info(
                f"Predictor ready — sequence_length={self._sequence_length}, "
                f"classes={self.config.get('class_names')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load predictor: {e}")
            return False

    def predict(
        self, feature_vector: np.ndarray, timestamp: float = 0.0
    ) -> Optional[RiskPrediction]:
        """
        Accept a new 1-second feature vector and return risk prediction.

        The buffer accumulates vectors until sequence_length is reached,
        then performs GRU inference. Subsequent calls update the rolling buffer.

        Args:
            feature_vector: 1D array of shape (n_features,).
            timestamp: Current time in seconds.

        Returns:
            RiskPrediction if buffer is full, None if still warming up.
        """
        if not self._is_loaded:
            logger.warning("Model not loaded — call load() first.")
            return None

        # Normalize the new feature vector
        fv_scaled = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()

        # Push to rolling buffer
        self._feature_buffer.append(fv_scaled)

        # Wait until buffer is full
        if len(self._feature_buffer) < self._sequence_length:
            remaining = self._sequence_length - len(self._feature_buffer)
            logger.debug(f"Buffer warming up — {remaining} more vectors needed.")
            return None

        # Build sequence input for GRU
        sequence = np.array(list(self._feature_buffer), dtype=np.float32)
        sequence = sequence.reshape(1, self._sequence_length, -1)

        # GRU inference
        raw_probs = self.model.predict(sequence, verbose=0)[0]  # Shape: (n_classes,)

        # Apply EMA smoothing to prevent flickering
        self._smoothed_probs = (
            self.ema_alpha * raw_probs +
            (1 - self.ema_alpha) * self._smoothed_probs
        )

        # Decode prediction
        pred_class_idx = np.argmax(self._smoothed_probs)
        pred_class = self.label_encoder.classes_[pred_class_idx]
        confidence = float(self._smoothed_probs[pred_class_idx])

        class_names = self.label_encoder.classes_.tolist()
        probabilities = {
            cls: float(self._smoothed_probs[i])
            for i, cls in enumerate(class_names)
        }

        prediction = RiskPrediction(
            risk_class=pred_class,
            probabilities=probabilities,
            confidence=confidence,
            timestamp=timestamp,
            smoothed=True,
        )

        self._prediction_history.append(prediction)
        return prediction

    def is_alert_condition(self, n_consecutive: int = 3) -> bool:
        """
        Check if a HIGH risk alert should be triggered.
        Requires n consecutive HIGH predictions to avoid false alarms.

        Args:
            n_consecutive: Number of consecutive HIGH predictions required.
        """
        if len(self._prediction_history) < n_consecutive:
            return False
        recent = list(self._prediction_history)[-n_consecutive:]
        return all(p.risk_class == "HIGH" for p in recent)

    def reset_buffer(self):
        """Reset temporal buffer (use when switching video sources)."""
        self._feature_buffer.clear()
        self._smoothed_probs = np.array([1.0, 0.0, 0.0])
        logger.info("Predictor buffer reset.")


# ─── Full Real-Time Inference Pipeline ───────────────────────────────────────

class RealtimeInferencePipeline:
    """
    Orchestrates the complete surveillance inference pipeline:
    Detection → Tracking → Analytics → GRU Prediction → Visualization

    This is the top-level class for production deployment.
    """

    def __init__(
        self,
        model_dir: str = "outputs/models",
        fps: float = 24.0,
        frame_width: int = 1280,
        frame_height: int = 720,
        detector_confidence: float = 0.40,
        show_visualization: bool = True,
    ):
        from detector import YOLOPersonDetector
        from tracker import CrowdTracker
        from analytics import CrowdAnalyticsEngine, FrameSnapshot

        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.show_visualization = show_visualization

        # Initialize pipeline components
        self.detector = YOLOPersonDetector(confidence_threshold=detector_confidence)
        self.tracker = CrowdTracker(fps=fps)
        self.analytics = CrowdAnalyticsEngine(fps=fps)
        self.predictor = GRURiskPredictor(model_dir=model_dir)

        # Runtime state
        self._frame_idx = 0
        self._current_time = 0.0
        self._current_risk = "UNKNOWN"
        self._current_probs = {}
        self._feature_vectors = []

        # Alert state
        self._alert_active = False
        self._alert_count = 0

        logger.info("RealtimeInferencePipeline initialized.")

    def initialize(self) -> bool:
        """Load all models and prepare for inference."""
        detector_ok = self.detector.load_model()
        predictor_ok = self.predictor.load()

        if not detector_ok:
            logger.warning("YOLO detector unavailable — using mock detections.")
        if not predictor_ok:
            logger.warning("GRU predictor unavailable — risk will show as UNKNOWN.")

        return True  # Continue with what's available

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[RiskPrediction]]:
        """
        Process one video frame through the complete pipeline.

        Args:
            frame: BGR image array.

        Returns:
            (annotated_frame, risk_prediction or None)
        """
        import cv2
        from analytics import FrameSnapshot

        self._frame_idx += 1
        self._current_time = self._frame_idx / self.fps

        # ── Detection ────────────────────────────────────────────────────────
        detections = self.detector.detect(frame)
        det_array = self.detector.detections_to_array(detections)

        # ── Tracking ─────────────────────────────────────────────────────────
        active_tracks = self.tracker.update(det_array, self._current_time)

        # ── Analytics (per-second aggregation) ───────────────────────────────
        snapshot = FrameSnapshot(
            frame_idx=self._frame_idx,
            timestamp=self._current_time,
            tracks=active_tracks,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        feature_vector = self.analytics.push_frame(snapshot)

        # ── GRU Prediction (once per second) ─────────────────────────────────
        prediction = None
        if feature_vector is not None:
            self._feature_vectors.append(feature_vector)
            fv_array = feature_vector.to_feature_array()
            prediction = self.predictor.predict(fv_array, self._current_time)

            if prediction:
                self._current_risk = prediction.risk_class
                self._current_probs = prediction.probabilities

            # Check alert condition
            if self.predictor.is_alert_condition(n_consecutive=3):
                if not self._alert_active:
                    self._alert_active = True
                    self._alert_count += 1
                    logger.warning(f"🚨 CONGESTION ALERT #{self._alert_count} — HIGH risk confirmed!")
            else:
                self._alert_active = False

        # ── Visualization ─────────────────────────────────────────────────────
        if self.show_visualization:
            from visualization import FrameAnnotator
            annotator = FrameAnnotator()
            frame = annotator.annotate_frame(
                frame, active_tracks, self._current_risk,
                self._current_probs, self._current_time,
                self._alert_active
            )

        return frame, prediction

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ):
        """
        Process a complete video file through the inference pipeline.

        Args:
            video_path: Path to input video.
            output_path: If provided, save annotated output video.
            max_frames: Limit processing to first N frames.
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        # Video properties
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Processing video: {video_path}\n"
            f"  Resolution: {width}×{height} @ {actual_fps:.1f}fps\n"
            f"  Total frames: {total}"
        )

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            annotated_frame, prediction = self.process_frame(frame)

            if writer:
                writer.write(annotated_frame)

            if self.show_visualization:
                cv2.imshow("FOOD_COURT_A — Crowd Risk Monitor", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Processing interrupted by user.")
                    break

            frame_count += 1

            if frame_count % 240 == 0:  # Log every 10 seconds
                logger.info(
                    f"Frame {frame_count}/{total} — "
                    f"Risk: {self._current_risk} — "
                    f"Alerts: {self._alert_count}"
                )

        cap.release()
        if writer:
            writer.release()
        if self.show_visualization:
            import cv2
            cv2.destroyAllWindows()

        logger.info(
            f"Video processing complete — "
            f"{frame_count} frames, {len(self._feature_vectors)} feature vectors"
        )

    def get_session_summary(self) -> Dict:
        """Return summary of the current inference session."""
        risk_counts = {"SAFE": 0, "WARNING": 0, "HIGH": 0}
        for fv in self._feature_vectors:
            risk_counts[fv.risk_label] = risk_counts.get(fv.risk_label, 0) + 1

        return {
            "total_frames": self._frame_idx,
            "total_seconds": len(self._feature_vectors),
            "risk_distribution": risk_counts,
            "total_alerts": self._alert_count,
            "alert_active": self._alert_active,
            "current_risk": self._current_risk,
        }
