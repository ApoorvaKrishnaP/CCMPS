"""
overlay.py — Real-time surveillance visualization overlays.

Renders on each frame:
  • Bounding boxes with tracking IDs (colour-coded by dwell time)
  • Trajectory trails (last N centroids)
  • Directional movement arrows
  • Zone risk status panel (SAFE / WARNING / HIGH)
  • Per-zone crowd density heatmap
  • Temporal risk confidence bar
  • Congestion alert banner for HIGH risk
  • People count + density readout
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

from tracker.deep_sort_tracker import TrackState
from utils.config import ZONES, ACTIVE_ZONE, RISK_CLASSES

# ── Colour constants (BGR) ──────────────────────────────────────────────────
_COLOUR_SAFE    = (50,  200,  50)
_COLOUR_WARNING = (30,  170, 230)
_COLOUR_HIGH    = (30,   30, 220)
_COLOUR_BOX_DEF = (200, 200, 200)
_COLOUR_TEXT    = (255, 255, 255)
_ALPHA_HEAT     = 0.35


def _risk_colour(label: str) -> Tuple[int, int, int]:
    return {"SAFE": _COLOUR_SAFE, "WARNING": _COLOUR_WARNING, "HIGH": _COLOUR_HIGH}.get(label, _COLOUR_SAFE)


def _dwell_colour(dwell_sec: float) -> Tuple[int, int, int]:
    """Colour tracks by how long the person has been present."""
    if dwell_sec < 30:   return (100, 255, 100)
    if dwell_sec < 120:  return (30,  200, 255)
    return (30, 30, 220)


# ── Main overlay class ────────────────────────────────────────────────────────

class SurveillanceOverlay:
    """Renders all visualization elements onto a BGR frame."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720) -> None:
        self._w = frame_width
        self._h = frame_height
        self._heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        self._heatmap_decay = 0.95

    def render(
        self,
        frame: np.ndarray,
        tracks: List[Tuple],                       # (tid, x1,y1,x2,y2,cx,cy)
        track_states: Dict[int, TrackState],
        risk_label: str = "SAFE",
        risk_probs: Optional[np.ndarray] = None,
        confidence: float = 1.0,
        second_idx: int = 0,
        is_ready: bool = True,
    ) -> np.ndarray:
        """
        Compose all overlays onto frame (in-place).

        Returns the modified frame.
        """
        overlay = frame.copy()

        # 1. Zone boundary
        self._draw_zone(overlay)

        # 2. Heatmap accumulation + blend
        self._update_heatmap(tracks)
        self._draw_heatmap(frame, overlay)

        # 3. Tracks: boxes, trails, arrows
        for (tid, x1, y1, x2, y2, cx, cy) in tracks:
            state = track_states.get(tid)
            dwell = state.dwell_time_sec() if state else 0.0
            colour = _dwell_colour(dwell)

            # Bounding box
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)

            # Track ID + dwell label
            label_txt = f"#{tid}  {dwell:.0f}s"
            cv2.putText(overlay, label_txt, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

            # Trajectory trail
            if state and len(state.centroids) > 1:
                pts = list(state.centroids)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    tc = tuple(int(c * alpha) for c in colour)
                    cv2.line(overlay, (int(pts[i-1][0]), int(pts[i-1][1])),
                             (int(pts[i][0]),   int(pts[i][1])), tc, 1)

            # Direction arrow
            if state:
                dirn = state.direction_deg()
                if dirn is not None:
                    rad = np.radians(dirn)
                    ex = int(cx + 20 * np.cos(rad))
                    ey = int(cy - 20 * np.sin(rad))
                    cv2.arrowedLine(overlay, (int(cx), int(cy)), (ex, ey),
                                   (255, 255, 0), 1, tipLength=0.4)

        # 4. Risk status panel (top-left)
        self._draw_risk_panel(overlay, risk_label, risk_probs, confidence, second_idx, is_ready, len(tracks))

        # 5. HIGH alert banner
        if risk_label == "HIGH":
            self._draw_alert_banner(overlay)

        return overlay

    # ── Panel drawing helpers ─────────────────────────────────────────────────

    def _draw_zone(self, frame: np.ndarray) -> None:
        x1, y1, x2, y2 = ZONES[ACTIVE_ZONE]["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 50), 1)
        cv2.putText(frame, ACTIVE_ZONE, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)

    def _update_heatmap(self, tracks: List[Tuple]) -> None:
        self._heatmap *= self._heatmap_decay
        for (_, x1, y1, x2, y2, cx, cy) in tracks:
            x, y = int(cx), int(cy)
            if 0 <= x < self._w and 0 <= y < self._h:
                cv2.circle(self._heatmap, (x, y), 30, 1.0, -1)

    def _draw_heatmap(self, base: np.ndarray, overlay: np.ndarray) -> None:
        hm_norm = cv2.normalize(self._heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hm_colour = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
        mask = (hm_norm > 10).astype(np.uint8)[:, :, np.newaxis]
        np.copyto(overlay, cv2.addWeighted(overlay, 1 - _ALPHA_HEAT, hm_colour, _ALPHA_HEAT, 0),
                  where=mask.repeat(3, axis=2).astype(bool))

    def _draw_risk_panel(
        self,
        frame: np.ndarray,
        label: str,
        probs: Optional[np.ndarray],
        confidence: float,
        second_idx: int,
        is_ready: bool,
        n_people: int,
    ) -> None:
        px, py = 10, 10
        bw, bh = 280, 160
        sub = frame[py:py+bh, px:px+bw]
        black = np.zeros_like(sub)
        cv2.addWeighted(black, 0.55, sub, 0.45, 0, sub)
        frame[py:py+bh, px:px+bw] = sub

        colour = _risk_colour(label)
        cv2.putText(frame, f"RISK: {label}", (px+8, py+32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, colour, 2)
        cv2.putText(frame, f"Conf: {confidence:.0%}   t={second_idx}s", (px+8, py+55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOUR_TEXT, 1)
        cv2.putText(frame, f"People: {n_people}", (px+8, py+75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOUR_TEXT, 1)

        if not is_ready:
            cv2.putText(frame, "Warming up...", (px+8, py+100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        if probs is not None:
            for ci, (cls, p) in enumerate(zip(RISK_CLASSES, probs)):
                bar_y = py + 110 + ci * 16
                bar_w = int(230 * float(p))
                cv2.rectangle(frame, (px+8, bar_y), (px+8+bar_w, bar_y+10), _risk_colour(cls), -1)
                cv2.putText(frame, f"{cls[:1]} {p:.2f}", (px+8+bar_w+4, bar_y+9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, _COLOUR_TEXT, 1)

    def _draw_alert_banner(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        banner_h = 50
        sub = frame[h-banner_h:h, 0:w]
        red_banner = np.full_like(sub, (30, 30, 200))
        cv2.addWeighted(red_banner, 0.7, sub, 0.3, 0, sub)
        frame[h-banner_h:h, 0:w] = sub
        cv2.putText(frame, "⚠  HIGH CONGESTION ALERT — FOOD_COURT_A  ⚠",
                    (w//2 - 310, h - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
