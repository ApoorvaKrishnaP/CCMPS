"""
realtime_predictor.py — GRU temporal predictor for live video inference.

TEMPORAL BUFFER DESIGN:
  The GRU requires a fixed-length sequence as input.
  We maintain a deque of the last SEQUENCE_LENGTH feature vectors.
  Once the buffer is full (after SEQUENCE_LENGTH seconds of video),
  we pass it to the GRU on every new second.

  This creates a sliding-window real-time prediction pipeline:
    buffer = [t-5, t-4, t-3, t-2, t-1]  →  predict risk at t
    buffer = [t-4, t-3, t-2, t-1, t  ]  →  predict risk at t+1
    ...

PREDICTION SMOOTHING:
  We apply a rolling majority vote over the last PREDICTION_SMOOTHING_WINDOW
  predictions to prevent flickering risk states (single-timestep anomalies
  causing rapid SAFE→HIGH→SAFE transitions that would confuse operators).
"""

from __future__ import annotations
import pickle
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.config import (
    SEQUENCE_LENGTH, FEATURE_COLUMNS, RISK_CLASSES,
    TRAINED_MODEL, SCALER_PATH, ENCODER_PATH,
    PREDICTION_SMOOTHING_WINDOW,
)


class RealTimePredictor:
    """
    Maintains temporal feature buffer and runs GRU inference each second.

    Usage:
        predictor = RealTimePredictor()

        # Called once per second with the feature dict from ZoneAnalytics
        label, probs, confidence = predictor.predict(feature_dict)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        encoder_path: Optional[Path] = None,
    ) -> None:
        mp = model_path   or TRAINED_MODEL
        sp = scaler_path  or SCALER_PATH
        ep = encoder_path or ENCODER_PATH

        self.model   = keras.models.load_model(str(mp))
        with open(sp, "rb") as f: self.scaler  = pickle.load(f)
        with open(ep, "rb") as f: self.encoder = pickle.load(f)

        self._buffer: deque = deque(maxlen=SEQUENCE_LENGTH)
        self._history: deque = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
        self._last_label = "SAFE"

        print(f"[Predictor] Loaded model from {mp}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        feature_dict: Dict[str, float],
    ) -> Tuple[str, np.ndarray, float]:
        """
        Ingest one second of crowd analytics and return risk prediction.

        Args:
            feature_dict: dict with keys matching FEATURE_COLUMNS.

        Returns:
            (label, probabilities, confidence)
              label:         "SAFE" | "WARNING" | "HIGH"
              probabilities: shape (3,) softmax output
              confidence:    max probability
        """
        vec = np.array([feature_dict.get(c, 0.0) for c in FEATURE_COLUMNS], dtype=np.float32)
        vec_scaled = self.scaler.transform(vec.reshape(1, -1))[0]
        self._buffer.append(vec_scaled)

        if len(self._buffer) < SEQUENCE_LENGTH:
            # Not enough history yet — return last known label
            probs = np.zeros(len(RISK_CLASSES))
            probs[RISK_CLASSES.index(self._last_label)] = 1.0
            return self._last_label, probs, 1.0

        seq = np.array(self._buffer)[np.newaxis, ...]   # (1, seq_len, features)
        probs = self.model.predict(seq, verbose=0)[0]   # (n_classes,)
        raw_label = RISK_CLASSES[int(np.argmax(probs))]

        # Smoothing: majority vote over recent predictions
        self._history.append(raw_label)
        smoothed_label = self._majority_vote()
        self._last_label = smoothed_label

        return smoothed_label, probs, float(np.max(probs))

    def is_ready(self) -> bool:
        """Returns True once the buffer has enough history for prediction."""
        return len(self._buffer) >= SEQUENCE_LENGTH

    def buffer_fill_pct(self) -> float:
        return len(self._buffer) / SEQUENCE_LENGTH

    # ── Internal ──────────────────────────────────────────────────────────────

    def _majority_vote(self) -> str:
        votes = list(self._history)
        return max(set(votes), key=votes.count)
