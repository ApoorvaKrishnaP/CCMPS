"""
deep_sort_tracker.py — DeepSORT-based persistent person tracker.

Why DeepSORT here?
  DeepSORT combines Kalman Filter motion prediction with a deep appearance
  descriptor (Re-ID feature). This gives persistent IDs even through partial
  occlusions — critical for accurate dwell-time and trajectory analysis.

Engineering notes:
  • We store a rolling trajectory deque for each track ID.
  • Trajectories older than TRAJECTORY_HISTORY_SEC are pruned to bound memory.
  • Centroid smoothing is applied before storing to reduce Kalman jitter.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from utils.config import (
    VIDEO_FPS, TRAJECTORY_HISTORY_SEC, MIN_TRACK_FRAMES, PIXEL_TO_METRE
)
import sys


class TrackState:
    """Per-track temporal state maintained across frames."""
    __slots__ = ("centroids", "first_seen_frame", "last_seen_frame", "bbox_history")

    def __init__(self, frame_idx: int, centroid: Tuple[float, float], bbox: Tuple) -> None:
        max_len = int(TRAJECTORY_HISTORY_SEC * VIDEO_FPS)
        self.centroids: deque = deque(maxlen=max_len)
        self.bbox_history: deque = deque(maxlen=max_len)
        self.first_seen_frame = frame_idx
        self.last_seen_frame  = frame_idx
        self.centroids.append(centroid)
        self.bbox_history.append(bbox)

    def update(self, frame_idx: int, centroid: Tuple[float, float], bbox: Tuple) -> None:
        self.centroids.append(centroid)
        self.bbox_history.append(bbox)
        self.last_seen_frame = frame_idx

    @property
    def age_frames(self) -> int:
        return self.last_seen_frame - self.first_seen_frame + 1

    def speed_mps(self, fps: int = VIDEO_FPS) -> float:
        """Instantaneous speed in metres/second from last two centroids."""
        if len(self.centroids) < 2:
            return 0.0
        c0 = np.array(self.centroids[-2])
        c1 = np.array(self.centroids[-1])
        dist_px = float(np.linalg.norm(c1 - c0))
        return dist_px * PIXEL_TO_METRE * fps

    def acceleration(self, fps: int = VIDEO_FPS) -> float:
        """Scalar acceleration from last three centroids."""
        if len(self.centroids) < 3:
            return 0.0
        pts = [np.array(c) for c in list(self.centroids)[-3:]]
        v0 = np.linalg.norm(pts[1] - pts[0]) * PIXEL_TO_METRE * fps
        v1 = np.linalg.norm(pts[2] - pts[1]) * PIXEL_TO_METRE * fps
        return float(v1 - v0)

    def direction_deg(self) -> Optional[float]:
        """Direction of travel in degrees (0=right, 90=up) from last 3 frames."""
        if len(self.centroids) < 2:
            return None
        c0 = np.array(self.centroids[-2])
        c1 = np.array(self.centroids[-1])
        dx, dy = c1 - c0
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return float(np.degrees(np.arctan2(-dy, dx)) % 360)  # image y is inverted

    def dwell_time_sec(self, fps: int = VIDEO_FPS) -> float:
        return self.age_frames / fps


class DeepSortTracker:
    """
    Thin wrapper around deep_sort_realtime providing trajectory bookkeeping.

    Usage:
        tracker = DeepSortTracker()
        for frame_idx, frame in enumerate(video):
            detections = yolo.detect(frame)
            tracks     = tracker.update(detections, frame, frame_idx)
            # tracks: list of (track_id, x1,y1,x2,y2, cx,cy)
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        embedder: str = "mobilenet",
    ) -> None:
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except Exception as e:  # pragma: no cover - runtime dependency failure
            msg = (
                "DeepSORT import failed: 'deep_sort_realtime' is required for "
                "`DeepSortTracker`.\n\nInstall into your active interpreter with:\n\n"
                f"  {sys.executable} -m pip install deep-sort-realtime opencv-python numpy\n\n"
                "If you intend to use appearance embedders (mobilenet) ensure "
                "PyTorch is installed for your Python version (Python 3.13 requires "
                "matching PyTorch wheels). On Windows, if installation fails for \"lap\" use:\n\n"
                f"  {sys.executable} -m pip install pipwin && pipwin install lap\n\n"
                "Alternatively, run the pipeline using the bundled `CrowdTracker` by "
                "setting the environment variable `USE_DEEPSORT=true` only after "
                "you've installed the requirements.\n\n"
                "Original import error: " + repr(e)
            )
            raise ImportError(msg)

        self._ds = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
            embedder=embedder,
            half=False,
        )
        self.track_states: Dict[int, TrackState] = {}
        self._frame_idx = 0
        print(f"[DeepSortTracker] Initialised | max_age={max_age} n_init={n_init}")

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Tuple],
        frame: np.ndarray,
        frame_idx: Optional[int] = None,
    ) -> List[Tuple]:
        """
        Args:
            detections: list of (x1,y1,x2,y2,conf) from YOLODetector.
            frame:      BGR numpy array for appearance extraction.
            frame_idx:  optional frame counter (auto-incremented if None).

        Returns:
            List of (track_id, x1, y1, x2, y2, cx, cy)
        """
        if frame_idx is None:
            frame_idx = self._frame_idx
        self._frame_idx = frame_idx + 1

        # deep_sort_realtime expects [[left,top,w,h], conf, cls]
        ds_input = []
        for (x1, y1, x2, y2, conf) in detections:
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], conf, 0))

        raw_tracks = self._ds.update_tracks(ds_input, frame=frame)

        active_ids = set()
        output = []

        for t in raw_tracks:
            if not t.is_confirmed():
                continue
            tid = int(t.track_id)
            x1, y1, x2, y2 = t.to_ltrb()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            active_ids.add(tid)

            if tid not in self.track_states:
                self.track_states[tid] = TrackState(frame_idx, (cx, cy), (x1,y1,x2,y2))
            else:
                self.track_states[tid].update(frame_idx, (cx, cy), (x1,y1,x2,y2))

            output.append((tid, float(x1), float(y1), float(x2), float(y2), cx, cy))

        # Prune stale states to prevent unbounded memory growth
        stale = [tid for tid in self.track_states if tid not in active_ids
                 and (frame_idx - self.track_states[tid].last_seen_frame) > VIDEO_FPS * 10]
        for tid in stale:
            del self.track_states[tid]

        return output

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self.track_states.get(track_id)

    def get_all_states(self) -> Dict[int, TrackState]:
        return self.track_states
