"""
tracker/deep_tracker.py
=======================
Persistent multi-object tracker for crowd surveillance.

Engineering Rationale:
- We implement a Kalman Filter + Hungarian Algorithm tracker inspired by
  DeepSORT/BoTSORT principles, without requiring GPU-based re-ID embeddings.
- For food court surveillance, appearance similarity matters less than
  spatial/motion consistency — so IoU + Kalman state is sufficient.
- Persistent IDs allow trajectory history accumulation across occlusions.
- The tracker maintains a 'grace period' before killing a track, handling
  temporary occlusions common in dense food court scenarios.

Track lifecycle:
    NEW → ACTIVE (after min_hits confirmations)
    ACTIVE → LOST (detection gap > max_age frames)
    LOST → DELETED (purged from memory)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


class KalmanBoxTracker:
    """
    Constant-velocity Kalman filter for bounding box tracking.

    State vector: [cx, cy, s, r, vx, vy, vs]
    - cx, cy: centroid position
    - s: bounding box scale (area)
    - r: aspect ratio (height/width, constant)
    - vx, vy: velocity components
    - vs: scale change rate

    Measurement vector: [cx, cy, s, r]

    Engineering Note:
    Kalman filtering smooths noisy YOLO detections and enables prediction
    of person position during brief occlusions — critical for maintaining
    trajectory continuity in crowded food court zones.
    """

    _id_counter = 0

    def __init__(self, bbox: np.ndarray, fps: float = 24.0):
        """
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2].
            fps: Video frame rate for velocity scaling.
        """
        KalmanBoxTracker._id_counter += 1
        self.track_id = KalmanBoxTracker._id_counter
        self.fps = fps
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.frames_since_update = 0
        self.time_since_update = 0

        # Trajectory history: list of (cx, cy, timestamp)
        self.trajectory: deque = deque(maxlen=60)  # Keep last 60 positions
        self.speed_history: deque = deque(maxlen=10)

        # Initialize Kalman filter
        self._init_kalman(bbox)

        # Store initial position
        cx, cy = self._get_centroid(bbox)
        self.trajectory.append((cx, cy, 0))

        logger.debug(f"New track created: ID={self.track_id}")

    def _init_kalman(self, bbox: np.ndarray):
        """Initialize Kalman filter matrices."""
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            # Fallback: simple position tracking without filterpy
            self._use_simple_tracking = True
            self._simple_bbox = bbox.copy()
            self._simple_velocity = np.zeros(4)
            return

        self._use_simple_tracking = False
        kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float32)

        # Measurement uncertainty
        kf.R[2:, 2:] *= 10.0

        # Initial state covariance
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0

        # Process noise
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        kf.x[:4] = self._bbox_to_z(bbox)
        self.kf = kf

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1,y1,x2,y2] to Kalman state [cx, cy, s, r]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h          # scale = area
        r = w / max(h, 1e-6)  # aspect ratio
        return np.array([[x], [y], [s], [r]], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """Convert Kalman state [cx, cy, s, r] back to [x1,y1,x2,y2]."""
        w = np.sqrt(max(z[2] * z[3], 0))
        h = max(z[2] / max(w, 1e-6), 0)
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _get_centroid(bbox: np.ndarray) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def predict(self) -> np.ndarray:
        """
        Advance Kalman state by one frame using motion model.
        Returns predicted bounding box [x1, y1, x2, y2].
        """
        self.age += 1
        self.frames_since_update += 1

        if self._use_simple_tracking:
            self._simple_bbox = self._simple_bbox + self._simple_velocity
            return self._simple_bbox

        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        return self._z_to_bbox(self.kf.x[:4].flatten())

    def update(self, bbox: np.ndarray, frame_time: float = 0.0):
        """
        Update track with a new matched detection.

        Args:
            bbox: Matched bounding box [x1, y1, x2, y2].
            frame_time: Current frame timestamp in seconds.

        Engineering Note:
        Speed is computed here from centroid displacement between consecutive
        updates. This is used downstream for congestion analytics.
        """
        self.hits += 1
        self.hit_streak += 1
        self.frames_since_update = 0

        # Update trajectory
        cx, cy = self._get_centroid(bbox)
        self.trajectory.append((cx, cy, frame_time))

        # Compute instantaneous speed if we have history
        if len(self.trajectory) >= 2:
            prev_cx, prev_cy, prev_t = self.trajectory[-2]
            dt = max(frame_time - prev_t, 1.0 / self.fps)
            dx = cx - prev_cx
            dy = cy - prev_cy
            pixel_speed = np.sqrt(dx**2 + dy**2) / dt
            self.speed_history.append(pixel_speed)

        if self._use_simple_tracking:
            old_bbox = self._simple_bbox.copy()
            self._simple_bbox = bbox
            self._simple_velocity = (bbox - old_bbox) * 0.1
            return

        self.kf.update(self._bbox_to_z(bbox))

    def get_state(self) -> np.ndarray:
        """Return current bounding box estimate."""
        if self._use_simple_tracking:
            return self._simple_bbox
        return self._z_to_bbox(self.kf.x[:4].flatten())

    @property
    def centroid(self) -> Tuple[float, float]:
        """Current centroid position."""
        bbox = self.get_state()
        return self._get_centroid(bbox)

    @property
    def avg_speed_pixels(self) -> float:
        """Average speed over recent history (pixels/second)."""
        if not self.speed_history:
            return 0.0
        return float(np.mean(list(self.speed_history)))

    @property
    def is_stagnant(self) -> bool:
        """True if the person has barely moved recently (stagnation detection)."""
        return self.avg_speed_pixels < 5.0  # < 5 pixels/sec = effectively standing still

    def get_direction_vector(self) -> Optional[np.ndarray]:
        """
        Compute recent movement direction vector.
        Returns normalized (dx, dy) or None if insufficient history.
        """
        if len(self.trajectory) < 3:
            return None

        recent = list(self.trajectory)[-5:]  # Last 5 positions
        if len(recent) < 2:
            return None

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        magnitude = np.sqrt(dx**2 + dy**2)

        if magnitude < 1e-6:
            return None

        return np.array([dx / magnitude, dy / magnitude])


class CrowdTracker:
    """
    Multi-object tracker managing all active person tracks.

    Architecture:
    - Maintains a pool of KalmanBoxTracker instances.
    - Each frame: predict all tracks → match detections via IoU + Hungarian.
    - Unmatched detections start new tentative tracks.
    - Unmatched tracks are aged; deleted after max_age frames.

    Engineering Note:
    The Hungarian algorithm (linear_sum_assignment) finds the globally
    optimal assignment between detections and tracks — better than greedy
    nearest-neighbor matching in dense crowds.
    """

    def __init__(
        self,
        max_age: int = 30,       # Frames before killing a lost track (30 = ~1.25 sec at 24fps)
        min_hits: int = 3,       # Confirmations before track is reported
        iou_threshold: float = 0.30,  # Min IoU for detection-track match
        fps: float = 24.0,
    ):
        """
        Args:
            max_age: Maximum frames a track can persist without detection update.
            min_hits: Minimum detections to confirm a tentative track.
            iou_threshold: Minimum IoU for valid association.
            fps: Source video frame rate.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.fps = fps
        self.tracks: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self._total_tracks_created = 0

        # Zone transition tracking
        self._prev_count = 0

        logger.info(
            f"CrowdTracker initialized — max_age={max_age}, "
            f"min_hits={min_hits}, iou_threshold={iou_threshold}"
        )

    def reset_id_counter(self):
        """Reset track ID counter (use between video sequences)."""
        KalmanBoxTracker._id_counter = 0

    @staticmethod
    def _compute_iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
        """
        Compute IoU matrix between two sets of bounding boxes.

        Args:
            bboxes_a: Shape (N, 4) — [x1, y1, x2, y2]
            bboxes_b: Shape (M, 4) — [x1, y1, x2, y2]

        Returns:
            IoU matrix of shape (N, M).
        """
        if len(bboxes_a) == 0 or len(bboxes_b) == 0:
            return np.zeros((len(bboxes_a), len(bboxes_b)))

        area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
        area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

        iou_matrix = np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)

        for i, a in enumerate(bboxes_a):
            xx1 = np.maximum(a[0], bboxes_b[:, 0])
            yy1 = np.maximum(a[1], bboxes_b[:, 1])
            xx2 = np.minimum(a[2], bboxes_b[:, 2])
            yy2 = np.minimum(a[3], bboxes_b[:, 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            union = area_a[i] + area_b - inter
            iou_matrix[i] = inter / np.maximum(union, 1e-6)

        return iou_matrix

    def _associate_detections(
        self, detections: np.ndarray, predicted_bboxes: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing tracks via IoU + Hungarian algorithm.

        Returns:
            matches: List of (track_idx, detection_idx) pairs.
            unmatched_tracks: Indices of tracks without matching detection.
            unmatched_detections: Indices of detections without matching track.
        """
        if len(predicted_bboxes) == 0:
            return [], [], list(range(len(detections)))

        if len(detections) == 0:
            return [], list(range(len(predicted_bboxes))), []

        iou_matrix = self._compute_iou_matrix(predicted_bboxes, detections)

        # Hungarian algorithm on cost matrix (1 - IoU)
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                if c in unmatched_detections:
                    unmatched_detections.remove(c)
            else:
                unmatched_tracks.append(r)

        # Tracks not covered by row_ind
        for r in range(len(predicted_bboxes)):
            if r not in [m[0] for m in matches] and r not in unmatched_tracks:
                unmatched_tracks.append(r)

        return matches, unmatched_tracks, unmatched_detections

    def update(
        self, detections: np.ndarray, frame_time: float = 0.0
    ) -> List[Dict]:
        """
        Update tracker with new detections from a single frame.

        Args:
            detections: Shape (N, 5) — [x1, y1, x2, y2, confidence].
                        Empty array if no detections.
            frame_time: Current frame timestamp in seconds.

        Returns:
            List of active track dicts with keys:
            - track_id, bbox, centroid, state, hits, speed_pixels,
              is_stagnant, trajectory, direction_vector
        """
        self.frame_count += 1

        # Step 1: Predict all tracks forward
        predicted_bboxes = []
        for track in self.tracks:
            pred = track.predict()
            predicted_bboxes.append(pred)

        # Step 2: Associate detections to predictions
        matches, unmatched_tracks, unmatched_detections = self._associate_detections(
            detections[:, :4] if len(detections) > 0 else np.empty((0, 4)),
            np.array(predicted_bboxes) if predicted_bboxes else np.empty((0, 4)),
        )

        # Step 3: Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx, :4], frame_time)

        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = KalmanBoxTracker(detections[det_idx, :4], fps=self.fps)
            self.tracks.append(new_track)
            self._total_tracks_created += 1

        # Step 5: Mark unmatched tracks as having missed an update
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].hit_streak = 0

        # Step 6: Confirm tentative tracks
        for track in self.tracks:
            if track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED

        # Step 7: Purge dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.frames_since_update <= self.max_age
        ]

        # Step 8: Build output for confirmed tracks only
        active_tracks = []
        for track in self.tracks:
            if track.state == TrackState.CONFIRMED and track.frames_since_update == 0:
                bbox = track.get_state()
                active_tracks.append({
                    "track_id": track.track_id,
                    "bbox": bbox,
                    "centroid": track.centroid,
                    "state": track.state,
                    "hits": track.hits,
                    "speed_pixels": track.avg_speed_pixels,
                    "is_stagnant": track.is_stagnant,
                    "trajectory": list(track.trajectory),
                    "direction_vector": track.get_direction_vector(),
                    "dwell_frames": track.hits,
                })

        return active_tracks

    def get_all_tracks(self) -> List[KalmanBoxTracker]:
        """Return all tracks including tentative and lost."""
        return self.tracks

    def get_stats(self) -> dict:
        """Runtime statistics for monitoring."""
        confirmed = sum(1 for t in self.tracks if t.state == TrackState.CONFIRMED)
        tentative = sum(1 for t in self.tracks if t.state == TrackState.TENTATIVE)
        return {
            "frame_count": self.frame_count,
            "active_confirmed": confirmed,
            "active_tentative": tentative,
            "total_created": self._total_tracks_created,
        }
