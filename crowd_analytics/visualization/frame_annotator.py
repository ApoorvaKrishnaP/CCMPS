"""
visualization/frame_annotator.py
=================================
Real-time surveillance visualization engine for FOOD_COURT_A.

Renders the following overlays on each video frame:
1. Bounding boxes with tracking IDs
2. Trajectory trails (recent path per person)
3. Movement direction arrows
4. Crowd density heatmap (background layer)
5. Zone risk status panel (top-right HUD)
6. Congestion alert banner (when HIGH)
7. Temporal risk probability bars
8. Frame statistics (FPS, count, timestamp)

Color Coding:
    Green  (#00C83C) — SAFE
    Orange (#FFA500) — WARNING
    Red    (#DC0000) — HIGH
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple


# ─── Color Scheme (BGR for OpenCV) ──────────────────────────────────────────
COLORS = {
    "SAFE":    (60, 200, 0),
    "WARNING": (0, 165, 255),
    "HIGH":    (0, 0, 220),
    "UNKNOWN": (150, 150, 150),
    "bbox_default": (100, 200, 100),
    "trajectory": (200, 200, 100),
    "arrow": (255, 200, 0),
    "hud_bg": (30, 30, 30),
    "text_white": (255, 255, 255),
    "text_gray": (180, 180, 180),
}

# Alert flash colors (for pulsing HIGH alert banner)
ALERT_COLORS = [(0, 0, 220), (0, 0, 160)]


class FrameAnnotator:
    """
    Renders all surveillance overlays onto a single video frame.

    All drawing operations use OpenCV primitives for maximum performance.
    The heatmap is computed from gaussian-blurred person centroids.
    """

    def __init__(
        self,
        show_trajectories: bool = True,
        show_heatmap: bool = True,
        show_arrows: bool = True,
        trajectory_length: int = 20,
    ):
        """
        Args:
            show_trajectories: Render trajectory trails behind persons.
            show_heatmap: Overlay gaussian crowd density heatmap.
            show_arrows: Draw movement direction arrows.
            trajectory_length: Number of trail points to display per person.
        """
        self.show_trajectories = show_trajectories
        self.show_heatmap = show_heatmap
        self.show_arrows = show_arrows
        self.trajectory_length = trajectory_length
        self._alert_frame_counter = 0

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        risk_class: str,
        probabilities: Dict[str, float],
        timestamp: float,
        alert_active: bool = False,
    ) -> np.ndarray:
        """
        Apply all visual overlays to a frame.

        Args:
            frame: BGR image to annotate (modified in place).
            tracks: Active track dictionaries from CrowdTracker.
            risk_class: Current risk classification ("SAFE", "WARNING", "HIGH").
            probabilities: Per-class probability dict.
            timestamp: Current video timestamp in seconds.
            alert_active: Whether HIGH alert is currently firing.

        Returns:
            Annotated frame (BGR).
        """
        # Work on a copy to preserve original
        out = frame.copy()
        h, w = out.shape[:2]

        # Layer 1: Crowd density heatmap (background)
        if self.show_heatmap and tracks:
            out = self._draw_heatmap(out, tracks)

        # Layer 2: Trajectory trails
        if self.show_trajectories:
            self._draw_trajectories(out, tracks)

        # Layer 3: Movement arrows
        if self.show_arrows:
            self._draw_direction_arrows(out, tracks)

        # Layer 4: Bounding boxes + Track IDs
        self._draw_bounding_boxes(out, tracks, risk_class)

        # Layer 5: HUD panel (top-right)
        self._draw_hud(out, risk_class, probabilities, len(tracks), timestamp)

        # Layer 6: Zone label
        self._draw_zone_label(out)

        # Layer 7: Alert banner (flashing, only when HIGH)
        if alert_active or risk_class == "HIGH":
            self._draw_alert_banner(out, risk_class)

        return out

    # ── Layer: Heatmap ───────────────────────────────────────────────────────

    def _draw_heatmap(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """
        Render crowd density heatmap from person centroid positions.

        Method: Gaussian kernel placed at each centroid, summed,
        normalized, then colorized with COLORMAP_JET and blended.
        """
        h, w = frame.shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)

        for track in tracks:
            cx, cy = track["centroid"]
            cx, cy = int(cx), int(cy)
            if 0 <= cx < w and 0 <= cy < h:
                # Gaussian blob centered at person
                y1 = max(0, cy - 40)
                y2 = min(h, cy + 40)
                x1 = max(0, cx - 30)
                x2 = min(w, cx + 30)
                heat[y1:y2, x1:x2] += 1.0

        if heat.max() > 0:
            heat_normalized = (heat / heat.max() * 255).astype(np.uint8)
            heat_blurred = cv2.GaussianBlur(heat_normalized, (51, 51), 0)
            heat_colored = cv2.applyColorMap(heat_blurred, cv2.COLORMAP_JET)
            mask = (heat_blurred > 20).astype(np.float32)
            alpha = (mask * 0.35)[..., np.newaxis]
            frame = (frame * (1 - alpha) + heat_colored * alpha).astype(np.uint8)

        return frame

    # ── Layer: Trajectories ──────────────────────────────────────────────────

    def _draw_trajectories(self, frame: np.ndarray, tracks: List[Dict]):
        """Draw fading trajectory trails for each tracked person."""
        for track in tracks:
            trajectory = track.get("trajectory", [])
            if len(trajectory) < 2:
                continue

            # Take recent N points
            recent = trajectory[-self.trajectory_length:]
            n_pts = len(recent)

            for i in range(1, n_pts):
                pt1 = (int(recent[i-1][0]), int(recent[i-1][1]))
                pt2 = (int(recent[i][0]), int(recent[i][1]))

                # Fade opacity: older points are more transparent
                alpha = i / n_pts
                color = (
                    int(COLORS["trajectory"][0] * alpha),
                    int(COLORS["trajectory"][1] * alpha),
                    int(COLORS["trajectory"][2] * alpha),
                )
                thickness = max(1, int(2 * alpha))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # ── Layer: Direction Arrows ──────────────────────────────────────────────

    def _draw_direction_arrows(self, frame: np.ndarray, tracks: List[Dict]):
        """Draw movement direction arrows from person centroids."""
        for track in tracks:
            dv = track.get("direction_vector")
            if dv is None:
                continue

            cx, cy = track["centroid"]
            cx, cy = int(cx), int(cy)

            speed = track.get("speed_pixels", 0.0)
            if speed < 3.0:  # Don't draw arrows for nearly-stationary persons
                continue

            arrow_len = min(25, max(10, int(speed * 0.3)))
            ex = int(cx + dv[0] * arrow_len)
            ey = int(cy + dv[1] * arrow_len)

            cv2.arrowedLine(
                frame, (cx, cy), (ex, ey),
                COLORS["arrow"], 2, cv2.LINE_AA, tipLength=0.35
            )

    # ── Layer: Bounding Boxes ────────────────────────────────────────────────

    def _draw_bounding_boxes(
        self, frame: np.ndarray, tracks: List[Dict], risk_class: str
    ):
        """
        Draw bounding boxes with track IDs.
        Box color depends on person's stagnation state:
        - Moving → green
        - Stagnant → orange/red (indicating potential congestion contribution)
        """
        for track in tracks:
            bbox = track["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            tid = track["track_id"]
            is_stagnant = track.get("is_stagnant", False)

            # Color: stagnant = orange, moving = green
            if is_stagnant:
                color = (0, 140, 255)  # Orange
            else:
                color = COLORS["bbox_default"]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Track ID label with background
            label = f"#{tid}"
            lx, ly = x1, max(y1 - 8, 12)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (lx - 1, ly - th - 3), (lx + tw + 1, ly + 2), color, -1)
            cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (20, 20, 20), 1, cv2.LINE_AA)

    # ── Layer: HUD Panel ─────────────────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        risk_class: str,
        probabilities: Dict[str, float],
        person_count: int,
        timestamp: float,
    ):
        """
        Draw information heads-up display panel in top-right corner.

        Displays:
        - Risk level indicator
        - Per-class probability bars
        - Person count
        - Timestamp
        """
        h, w = frame.shape[:2]
        panel_w = 210
        panel_h = 185
        px = w - panel_w - 10
        py = 10

        # Panel background (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h),
                      COLORS["hud_bg"], -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h),
                      COLORS.get(risk_class, COLORS["UNKNOWN"]), 2)

        # Risk level title
        risk_color = COLORS.get(risk_class, COLORS["UNKNOWN"])
        cv2.putText(frame, "FOOD_COURT_A",
                    (px + 10, py + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, COLORS["text_gray"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"RISK: {risk_class}",
                    (px + 10, py + 42), cv2.FONT_HERSHEY_DUPLEX,
                    0.65, risk_color, 2, cv2.LINE_AA)

        # Divider
        cv2.line(frame, (px + 8, py + 50), (px + panel_w - 8, py + 50),
                 (80, 80, 80), 1)

        # Probability bars
        bar_y = py + 62
        classes_order = ["SAFE", "WARNING", "HIGH"]
        bar_colors = [COLORS["SAFE"], COLORS["WARNING"], COLORS["HIGH"]]
        bar_w = 110

        for cls, col in zip(classes_order, bar_colors):
            prob = probabilities.get(cls, 0.0)
            fill = int(bar_w * prob)

            cv2.putText(frame, cls[:4], (px + 10, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["text_gray"],
                        1, cv2.LINE_AA)
            # Background bar
            cv2.rectangle(frame, (px + 50, bar_y), (px + 50 + bar_w, bar_y + 12),
                          (60, 60, 60), -1)
            # Filled bar
            if fill > 0:
                cv2.rectangle(frame, (px + 50, bar_y), (px + 50 + fill, bar_y + 12),
                              col, -1)
            # Percentage
            cv2.putText(frame, f"{prob:.0%}",
                        (px + 50 + bar_w + 4, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["text_white"],
                        1, cv2.LINE_AA)
            bar_y += 25

        # Divider
        cv2.line(frame, (px + 8, bar_y + 3), (px + panel_w - 8, bar_y + 3),
                 (80, 80, 80), 1)

        # Person count + timestamp
        cv2.putText(frame, f"Count: {person_count}",
                    (px + 10, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, COLORS["text_white"], 1, cv2.LINE_AA)
        ts_str = f"T: {timestamp:.1f}s"
        cv2.putText(frame, ts_str,
                    (px + 10, bar_y + 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, COLORS["text_gray"], 1, cv2.LINE_AA)

    # ── Layer: Zone Label ────────────────────────────────────────────────────

    def _draw_zone_label(self, frame: np.ndarray):
        """Draw monitored zone name in top-left corner."""
        h, w = frame.shape[:2]
        label = "[ ZONE: FOOD_COURT_A ]"
        cv2.putText(frame, label, (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLORS["text_white"],
                    1, cv2.LINE_AA)
        cv2.putText(frame, "AI SURVEILLANCE ACTIVE", (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 100),
                    1, cv2.LINE_AA)

        # Recording dot
        self._alert_frame_counter += 1
        if (self._alert_frame_counter // 12) % 2 == 0:
            cv2.circle(frame, (w - 30, 20), 7, (0, 0, 200), -1)
            cv2.putText(frame, "REC", (w - 65, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200),
                        1, cv2.LINE_AA)

    # ── Layer: Alert Banner ──────────────────────────────────────────────────

    def _draw_alert_banner(self, frame: np.ndarray, risk_class: str):
        """
        Draw flashing alert banner at bottom of frame.
        Pulses between two shades of red for HIGH risk.
        """
        h, w = frame.shape[:2]
        banner_h = 50

        # Alternating flash
        flash_idx = (self._alert_frame_counter // 8) % 2

        if risk_class == "HIGH":
            bg_color = ALERT_COLORS[flash_idx]
            text = "🚨  CONGESTION ALERT — HIGH RISK — FOOD_COURT_A  🚨"
            text_color = (255, 255, 255)
        elif risk_class == "WARNING":
            bg_color = (0, 100, 200)
            text = "⚠  WARNING — ELEVATED CROWD DENSITY DETECTED  ⚠"
            text_color = (255, 255, 200)
        else:
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Center text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.62, 2)
        tx = max(10, (w - tw) // 2)
        cv2.putText(frame, text, (tx, h - 17),
                    cv2.FONT_HERSHEY_DUPLEX, 0.62, text_color, 2, cv2.LINE_AA)
