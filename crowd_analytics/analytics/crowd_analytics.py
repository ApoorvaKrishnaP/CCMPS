"""
analytics/crowd_analytics.py
=============================
Temporal crowd analytics engine for FOOD_COURT_A.

Engineering Rationale:
- Per-second aggregation (not per-frame) gives the GRU stable temporal patterns.
- At 24 FPS, each second produces 1 feature vector from 24 frames of tracking data.
- This temporal smoothing suppresses detection noise and tracking jitter.
- The analytics are zone-specific (FOOD_COURT_A) for focused congestion modeling.

Why per-second aggregation matters:
    Frame-level metrics oscillate wildly due to detection noise.
    Per-second aggregation captures the *behavioral trend* the GRU needs to learn.
    Congestion builds over seconds/minutes, not individual frames.

Feature Definitions:
    people_count        — average tracked persons in the zone per second
    density             — people_count normalized by zone area
    avg_speed_mps       — average movement speed (pixel→meter converted)
    stagnation_ratio    — fraction of tracked persons standing still
    avg_dwell_time_sec  — how long persons have been in the zone on average
    flow_conflict_ratio — fraction of opposing movement vectors
    directional_entropy — Shannon entropy of movement direction histogram
    acceleration_variance — variance in speed changes (sudden stops = congestion)
    zone_transitions    — estimated entries + exits per second
    inflow              — new persons entering the zone per second
    outflow             — persons leaving the zone per second
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import math
import logging

logger = logging.getLogger(__name__)


# ─── Zone Configuration ─────────────────────────────────────────────────────

ZONE_FOOD_COURT_A = {
    "name": "FOOD_COURT_A",
    "area_m2": 120.0,           # Approximate zone area in square meters
    "pixels_per_meter": 20.0,   # Camera calibration: pixels per meter
    "max_capacity": 150,        # Maximum safe occupancy
    "warning_density": 0.5,     # Density threshold for WARNING risk
    "high_density": 0.75,       # Density threshold for HIGH risk
}


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class FrameSnapshot:
    """Single-frame tracking snapshot passed to analytics buffer."""
    frame_idx: int
    timestamp: float
    tracks: List[Dict]          # Active track dicts from CrowdTracker.update()
    frame_width: int = 1280
    frame_height: int = 720


@dataclass
class CrowdFeatureVector:
    """
    One second of aggregated crowd analytics for FOOD_COURT_A.
    This is the fundamental unit fed to the GRU model.
    """
    timestamp: float
    zone_name: str = "FOOD_COURT_A"

    # Core occupancy
    people_count: float = 0.0
    density: float = 0.0

    # Movement analytics
    avg_speed_mps: float = 0.0
    stagnation_ratio: float = 0.0
    avg_dwell_time_sec: float = 0.0
    flow_conflict_ratio: float = 0.0
    directional_entropy: float = 0.0
    acceleration_variance: float = 0.0

    # Flow analytics
    zone_transitions: float = 0.0
    inflow: float = 0.0
    outflow: float = 0.0

    # Risk label (assigned externally)
    risk_label: str = "UNKNOWN"

    def to_feature_array(self) -> np.ndarray:
        """Return numeric features as numpy array for ML pipeline."""
        return np.array([
            self.people_count,
            self.density,
            self.avg_speed_mps,
            self.stagnation_ratio,
            self.avg_dwell_time_sec,
            self.flow_conflict_ratio,
            self.directional_entropy,
            self.acceleration_variance,
            self.zone_transitions,
            self.inflow,
            self.outflow,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        """Serialize to dictionary for CSV export."""
        return {
            "timestamp": self.timestamp,
            "zone_name": self.zone_name,
            "people_count": round(self.people_count, 2),
            "density": round(self.density, 4),
            "avg_speed_mps": round(self.avg_speed_mps, 4),
            "stagnation_ratio": round(self.stagnation_ratio, 4),
            "avg_dwell_time_sec": round(self.avg_dwell_time_sec, 2),
            "flow_conflict_ratio": round(self.flow_conflict_ratio, 4),
            "directional_entropy": round(self.directional_entropy, 4),
            "acceleration_variance": round(self.acceleration_variance, 6),
            "zone_transitions": round(self.zone_transitions, 2),
            "inflow": round(self.inflow, 2),
            "outflow": round(self.outflow, 2),
            "risk_label": self.risk_label,
        }

    FEATURE_NAMES = [
        "people_count", "density", "avg_speed_mps", "stagnation_ratio",
        "avg_dwell_time_sec", "flow_conflict_ratio", "directional_entropy",
        "acceleration_variance", "zone_transitions", "inflow", "outflow",
    ]


# ─── Analytics Engine ────────────────────────────────────────────────────────

class CrowdAnalyticsEngine:
    """
    Aggregates per-frame tracking data into per-second crowd feature vectors.

    Usage:
        engine = CrowdAnalyticsEngine(fps=24)
        for frame in video:
            snapshot = FrameSnapshot(...)
            fv = engine.push_frame(snapshot)
            if fv is not None:
                # One second has elapsed — fv is ready for GRU
                dataset.append(fv)
    """

    NUM_DIRECTION_BINS = 8      # 8-bin direction histogram (45° each)
    STAGNATION_SPEED_THRESHOLD = 0.05   # m/s — below this = stagnant

    def __init__(
        self,
        fps: float = 24.0,
        zone_config: dict = ZONE_FOOD_COURT_A,
        aggregation_window: int = 24,   # Frames per feature vector (1 second)
    ):
        """
        Args:
            fps: Source video frame rate.
            zone_config: Zone metadata dictionary.
            aggregation_window: Frames to aggregate per feature vector.
        """
        self.fps = fps
        self.zone = zone_config
        self.aggregation_window = aggregation_window
        self.pixels_per_meter = zone_config["pixels_per_meter"]

        # Frame buffer: accumulates snapshots for 1 second
        self._frame_buffer: List[FrameSnapshot] = []

        # Track ID sets for inflow/outflow computation
        self._prev_track_ids: set = set()
        self._current_second = 0

        # Speed history per track for acceleration computation
        self._track_speed_history: Dict[int, deque] = {}

        # Dwell time tracking: track_id → first_seen_timestamp
        self._track_first_seen: Dict[int, float] = {}

        logger.info(
            f"CrowdAnalyticsEngine initialized — fps={fps}, "
            f"zone={zone_config['name']}, window={aggregation_window} frames"
        )

    def push_frame(self, snapshot: FrameSnapshot) -> Optional[CrowdFeatureVector]:
        """
        Push one frame's tracking data into the buffer.

        Returns:
            CrowdFeatureVector if the 1-second window is complete, else None.

        Engineering Note:
        This function implements the critical "aggregate over 1 second" design.
        The GRU never sees individual frames — only 1-second summaries.
        """
        self._frame_buffer.append(snapshot)

        # Update dwell time registry
        for track in snapshot.tracks:
            tid = track["track_id"]
            if tid not in self._track_first_seen:
                self._track_first_seen[tid] = snapshot.timestamp

        if len(self._frame_buffer) >= self.aggregation_window:
            fv = self._compute_feature_vector()
            self._frame_buffer = []  # Reset buffer
            self._current_second += 1
            return fv

        return None

    def _compute_feature_vector(self) -> CrowdFeatureVector:
        """
        Compute all 11 temporal features from the accumulated frame buffer.
        This is the core analytics computation.
        """
        if not self._frame_buffer:
            return self._empty_vector()

        ts = self._frame_buffer[-1].timestamp
        frame_w = self._frame_buffer[-1].frame_width
        frame_h = self._frame_buffer[-1].frame_height
        zone_area_pixels = frame_w * frame_h  # Treat full frame as zone

        # ── Collect all track observations across the 1-second window ────────
        all_tracks: Dict[int, List[Dict]] = {}
        for snap in self._frame_buffer:
            for t in snap.tracks:
                tid = t["track_id"]
                if tid not in all_tracks:
                    all_tracks[tid] = []
                all_tracks[tid].append(t)

        current_track_ids = set(all_tracks.keys())
        unique_ids = current_track_ids

        # ── Feature 1: People Count ──────────────────────────────────────────
        counts_per_frame = [len(snap.tracks) for snap in self._frame_buffer]
        people_count = float(np.mean(counts_per_frame)) if counts_per_frame else 0.0

        # ── Feature 2: Density ───────────────────────────────────────────────
        # Normalize by zone area capacity
        density = min(people_count / max(self.zone["max_capacity"], 1), 1.0)

        # ── Feature 3: Average Speed (m/s) ───────────────────────────────────
        speeds_mps = []
        for tid, observations in all_tracks.items():
            for obs in observations:
                pixel_speed = obs.get("speed_pixels", 0.0)
                mps = pixel_speed / self.pixels_per_meter
                speeds_mps.append(mps)
        avg_speed_mps = float(np.mean(speeds_mps)) if speeds_mps else 0.0

        # ── Feature 4: Stagnation Ratio ──────────────────────────────────────
        # Fraction of persons moving below stagnation threshold
        stagnant_count = sum(
            1 for tid, obs_list in all_tracks.items()
            if np.mean([o.get("speed_pixels", 0) for o in obs_list]) / self.pixels_per_meter
            < self.STAGNATION_SPEED_THRESHOLD
        )
        stagnation_ratio = stagnant_count / max(len(unique_ids), 1)

        # ── Feature 5: Average Dwell Time ────────────────────────────────────
        dwell_times = []
        for tid in unique_ids:
            first_seen = self._track_first_seen.get(tid, ts)
            dwell = ts - first_seen
            dwell_times.append(max(dwell, 0.0))
        avg_dwell_time_sec = float(np.mean(dwell_times)) if dwell_times else 0.0

        # ── Feature 6: Flow Conflict Ratio ───────────────────────────────────
        # Fraction of pairs with opposing movement vectors (cosine < -0.5)
        direction_vectors = []
        for tid, obs_list in all_tracks.items():
            for obs in obs_list:
                dv = obs.get("direction_vector")
                if dv is not None and np.linalg.norm(dv) > 0.1:
                    direction_vectors.append(dv)

        flow_conflict_ratio = self._compute_flow_conflict(direction_vectors)

        # ── Feature 7: Directional Entropy ───────────────────────────────────
        directional_entropy = self._compute_directional_entropy(direction_vectors)

        # ── Feature 8: Acceleration Variance ─────────────────────────────────
        # High variance = sudden stops/starts = congestion signal
        acc_values = []
        for tid, obs_list in all_tracks.items():
            speeds = [o.get("speed_pixels", 0) / self.pixels_per_meter for o in obs_list]
            if len(speeds) >= 2:
                accs = np.diff(speeds)
                acc_values.extend(accs.tolist())
        acceleration_variance = float(np.var(acc_values)) if acc_values else 0.0

        # ── Feature 9: Zone Transitions ──────────────────────────────────────
        new_ids = current_track_ids - self._prev_track_ids
        gone_ids = self._prev_track_ids - current_track_ids
        inflow = float(len(new_ids))
        outflow = float(len(gone_ids))
        zone_transitions = inflow + outflow

        # Update ID history for next second
        self._prev_track_ids = current_track_ids

        # ── Assemble Feature Vector ───────────────────────────────────────────
        fv = CrowdFeatureVector(
            timestamp=ts,
            people_count=people_count,
            density=density,
            avg_speed_mps=avg_speed_mps,
            stagnation_ratio=stagnation_ratio,
            avg_dwell_time_sec=avg_dwell_time_sec,
            flow_conflict_ratio=flow_conflict_ratio,
            directional_entropy=directional_entropy,
            acceleration_variance=acceleration_variance,
            zone_transitions=zone_transitions,
            inflow=inflow,
            outflow=outflow,
        )

        # ── Auto-label risk based on density + stagnation heuristics ─────────
        fv.risk_label = self._auto_label_risk(fv)

        return fv

    def _compute_flow_conflict(self, direction_vectors: List[np.ndarray]) -> float:
        """
        Compute fraction of pairs with opposing movement directions.

        Engineering Note:
        Flow conflict is a key congestion signal: when people move against
        each other (e.g., incoming vs. outgoing queues), it creates bottlenecks.
        We sample pairs to avoid O(N²) complexity at high density.
        """
        if len(direction_vectors) < 2:
            return 0.0

        # Sample up to 50 pairs for efficiency
        n = len(direction_vectors)
        conflicts = 0
        pairs_checked = 0
        max_pairs = min(n * (n - 1) // 2, 50)

        indices = np.random.choice(n, size=min(n, 15), replace=False)
        vectors = [direction_vectors[i] for i in indices]

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                cos_sim = np.dot(vectors[i], vectors[j])
                if cos_sim < -0.5:  # Opposing direction
                    conflicts += 1
                pairs_checked += 1
                if pairs_checked >= max_pairs:
                    break
            if pairs_checked >= max_pairs:
                break

        return conflicts / max(pairs_checked, 1)

    def _compute_directional_entropy(self, direction_vectors: List[np.ndarray]) -> float:
        """
        Shannon entropy of 8-bin movement direction histogram.

        High entropy = chaotic movement (all directions equally likely).
        Low entropy = organized flow (everyone moving same direction).

        Range: [0, log(8)] ≈ [0, 2.08] — normalized to [0, 1].
        """
        if not direction_vectors:
            return 0.0

        # Bin directions into 8 octants
        histogram = np.zeros(self.NUM_DIRECTION_BINS, dtype=float)
        for dv in direction_vectors:
            angle = math.atan2(dv[1], dv[0])  # [-π, π]
            bin_idx = int((angle + math.pi) / (2 * math.pi) * self.NUM_DIRECTION_BINS)
            bin_idx = min(bin_idx, self.NUM_DIRECTION_BINS - 1)
            histogram[bin_idx] += 1

        # Normalize to probability distribution
        total = histogram.sum()
        if total < 1e-6:
            return 0.0
        probs = histogram / total

        # Shannon entropy
        entropy = -np.sum(p * math.log(p + 1e-9) for p in probs)
        max_entropy = math.log(self.NUM_DIRECTION_BINS)

        return float(entropy / max_entropy)  # Normalized to [0, 1]

    def _auto_label_risk(self, fv: CrowdFeatureVector) -> str:
        """
        Heuristic-based risk labeling for training data generation.

        Labels:
            SAFE    — Normal occupancy, good movement flow
            WARNING — Elevated density or stagnation beginning
            HIGH    — Dangerous congestion, movement breakdown

        Engineering Note:
        These rules are domain-specific for food courts. The GRU learns
        to anticipate these transitions BEFORE they become visible.
        """
        density = fv.density
        stagnation = fv.stagnation_ratio
        speed = fv.avg_speed_mps
        conflict = fv.flow_conflict_ratio

        # HIGH risk conditions
        if (density > self.zone["high_density"]
                or (stagnation > 0.7 and density > 0.5)
                or (conflict > 0.6 and density > 0.55)
                or (speed < 0.1 and density > 0.6)):
            return "HIGH"

        # WARNING conditions
        elif (density > self.zone["warning_density"]
              or stagnation > 0.45
              or conflict > 0.40
              or (fv.directional_entropy > 0.75 and density > 0.35)):
            return "WARNING"

        # SAFE
        else:
            return "SAFE"

    def _empty_vector(self) -> CrowdFeatureVector:
        """Return a zero-filled feature vector."""
        return CrowdFeatureVector(timestamp=0.0, risk_label="SAFE")

    def get_current_buffer_size(self) -> int:
        """Return number of frames currently buffered."""
        return len(self._frame_buffer)
