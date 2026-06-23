"""
trajectory_analytics.py — Temporal crowd behaviour analytics engine.

KEY DESIGN PRINCIPLE:
  We aggregate metrics over 1-second windows (24 frames @ 24 FPS).
  This produces one feature vector per second — the temporal unit the GRU learns from.
  Frame-level analytics would be too noisy and computationally wasteful.

Computed features (see FEATURE_COLUMNS in config.py):
  ┌─────────────────────────┬────────────────────────────────────────────────┐
  │ Feature                 │ Engineering rationale                          │
  ├─────────────────────────┼────────────────────────────────────────────────┤
  │ people_count            │ Raw occupancy signal                           │
  │ density                 │ people_count / zone_area_m2                    │
  │ avg_speed_mps           │ Mean pedestrian speed across active tracks     │
  │ stagnation_ratio        │ Fraction of people below STAGNATION threshold  │
  │ avg_dwell_time_sec      │ Mean time-in-zone per tracked person           │
  │ flow_conflict_ratio     │ Fraction of track pairs with opposing vectors  │
  │ directional_entropy     │ Shannon entropy of direction histogram         │
  │ acceleration_variance   │ Variance of scalar accelerations               │
  │ zone_transitions        │ Counts of new entries + exits per second       │
  │ inflow                  │ New track IDs seen this second                 │
  │ outflow                 │ Track IDs that left this second                │
  └─────────────────────────┴────────────────────────────────────────────────┘
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Set, Tuple
from scipy.stats import entropy as scipy_entropy

from utils.config import (
    ZONES, ACTIVE_ZONE,
    STAGNATION_SPEED_THRESH_MPS,
    MIN_TRACK_FRAMES,
    VIDEO_FPS,
)
from tracker.deep_sort_tracker import TrackState


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _directional_entropy(directions_deg: List[float], n_bins: int = 8) -> float:
    """
    Shannon entropy of the direction histogram.

    High entropy → chaotic, multi-directional crowd (typical of high congestion).
    Low entropy  → orderly flow (exit/entry queue behaviour).
    """
    if len(directions_deg) == 0:
        return 0.0
    hist, _ = np.histogram(directions_deg, bins=n_bins, range=(0, 360))
    hist = hist.astype(float)
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist / total
    # scipy_entropy uses natural log; normalise to [0, 1] by dividing by log(n_bins)
    raw = float(scipy_entropy(prob + 1e-9))
    return min(raw / np.log(n_bins), 1.0)


def _flow_conflict_ratio(directions_deg: List[float], angle_thresh: float = 120.0) -> float:
    """
    Fraction of direction pairs whose angular difference > angle_thresh.

    Opposing pedestrian flows create friction that precedes congestion.
    """
    if len(directions_deg) < 2:
        return 0.0
    dirs = np.array(directions_deg)
    conflicts = 0
    total = 0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            diff = abs(dirs[i] - dirs[j])
            diff = min(diff, 360 - diff)
            if diff > angle_thresh:
                conflicts += 1
            total += 1
    return conflicts / max(total, 1)


# ─── Main Analytics Class ────────────────────────────────────────────────────────

class ZoneAnalytics:
    """
    Aggregates per-frame tracking data into 1-second feature vectors.

    Usage pattern:
        analytics = ZoneAnalytics()

        # Inside frame loop (call at 24 FPS):
        analytics.ingest_frame(active_tracks, all_track_states, frame_idx)

        # Once per second (when frame_idx % VIDEO_FPS == 0):
        feature_vector = analytics.flush_window()
    """

    def __init__(self) -> None:
        self._zone_cfg   = ZONES[ACTIVE_ZONE]
        self._area_m2    = self._zone_cfg["area_m2"]
        self._zone_bbox  = self._zone_cfg["bbox"]

        # Rolling buffers for the current 1-second window
        self._reset_buffers()

        # Track IDs present at the start of the previous window (for inflow/outflow)
        self._prev_ids: Set[int] = set()

    # ── Internal ────────────────────────────────────────────────────────────────

    def _reset_buffers(self) -> None:
        self._frame_counts:    List[int]   = []   # people per frame
        self._speeds:          List[float] = []   # per-track instantaneous speeds
        self._accelerations:   List[float] = []
        self._directions:      List[float] = []
        self._dwell_times:     List[float] = []
        self._stagnant_counts: List[int]   = []
        self._active_counts:   List[int]   = []
        self._current_ids:     Set[int]    = set()

        # Per-frame track ID sets for inflow/outflow delta
        self._frame_id_sets:   List[Set[int]] = []

    def _in_zone(self, cx: float, cy: float) -> bool:
        x1, y1, x2, y2 = self._zone_bbox
        return x1 <= cx <= x2 and y1 <= cy <= y2

    # ── Public API ───────────────────────────────────────────────────────────────

    def ingest_frame(
        self,
        active_tracks: List[Tuple],          # (tid, x1,y1,x2,y2, cx, cy)
        track_states:  Dict[int, TrackState],
        frame_idx: int,
    ) -> None:
        """
        Accumulate per-frame data into the current 1-second window buffer.
        Called once per frame inside the main video loop.
        """
        in_zone = [(tid, cx, cy) for (tid, _, _, _, _, cx, cy) in active_tracks
                   if self._in_zone(cx, cy)]

        frame_ids = set(tid for (tid, _, _) in in_zone)
        self._frame_id_sets.append(frame_ids)
        self._current_ids |= frame_ids
        self._frame_counts.append(len(in_zone))

        stagnant = 0
        for (tid, cx, cy) in in_zone:
            state = track_states.get(tid)
            if state is None or state.age_frames < MIN_TRACK_FRAMES:
                continue
            spd = state.speed_mps()
            acc = state.acceleration()
            dirn = state.direction_deg()

            self._speeds.append(spd)
            self._accelerations.append(acc)
            if dirn is not None:
                self._directions.append(dirn)
            self._dwell_times.append(state.dwell_time_sec())

            if spd < STAGNATION_SPEED_THRESH_MPS:
                stagnant += 1

        self._stagnant_counts.append(stagnant)
        self._active_counts.append(len(in_zone))

    def flush_window(self) -> Dict[str, float]:
        """
        Compute and return the 1-second feature vector, then reset buffers.

        Returns:
            dict matching FEATURE_COLUMNS in config.py
        """
        # ── People Count ────────────────────────────────────────────────────────
        people_count = float(np.mean(self._frame_counts)) if self._frame_counts else 0.0

        # ── Density (persons per m²) ─────────────────────────────────────────────
        density = people_count / self._area_m2

        # ── Speed ────────────────────────────────────────────────────────────────
        avg_speed_mps = float(np.mean(self._speeds)) if self._speeds else 0.0

        # ── Stagnation Ratio ─────────────────────────────────────────────────────
        total_presences = sum(self._active_counts)
        total_stagnant  = sum(self._stagnant_counts)
        stagnation_ratio = total_stagnant / max(total_presences, 1)

        # ── Dwell Time ───────────────────────────────────────────────────────────
        avg_dwell_time_sec = float(np.mean(self._dwell_times)) if self._dwell_times else 0.0

        # ── Flow Conflict ────────────────────────────────────────────────────────
        flow_conflict_ratio = _flow_conflict_ratio(self._directions)

        # ── Directional Entropy ──────────────────────────────────────────────────
        directional_entropy = _directional_entropy(self._directions)

        # ── Acceleration Variance ────────────────────────────────────────────────
        acceleration_variance = float(np.var(self._accelerations)) if self._accelerations else 0.0

        # ── Inflow / Outflow ─────────────────────────────────────────────────────
        # inflow  = IDs active this window but NOT in previous window
        # outflow = IDs in previous window but NOT active this window
        inflow  = float(len(self._current_ids - self._prev_ids))
        outflow = float(len(self._prev_ids - self._current_ids))

        # ── Zone Transitions (inflow + outflow) ──────────────────────────────────
        zone_transitions = inflow + outflow

        # Save current IDs as previous for next window
        self._prev_ids = set(self._current_ids)
        self._reset_buffers()

        return {
            "people_count":         people_count,
            "density":              density,
            "avg_speed_mps":        avg_speed_mps,
            "stagnation_ratio":     stagnation_ratio,
            "avg_dwell_time_sec":   avg_dwell_time_sec,
            "flow_conflict_ratio":  flow_conflict_ratio,
            "directional_entropy":  directional_entropy,
            "acceleration_variance": acceleration_variance,
            "zone_transitions":     zone_transitions,
            "inflow":               inflow,
            "outflow":              outflow,
        }
