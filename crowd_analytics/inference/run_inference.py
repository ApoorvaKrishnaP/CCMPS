"""
run_inference.py — Real-time crowd risk inference on live or recorded video.

Full pipeline:
  Video → YOLO → DeepSORT → ZoneAnalytics (1s window) → GRU → Risk Overlay → Output

Usage examples:
  # Process recorded video
  python inference/run_inference.py --source mall_footage.mp4

  # Live webcam
  python inference/run_inference.py --source 0

  # With output video saved
  python inference/run_inference.py --source mall_footage.mp4 --output outputs/result.mp4
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector.yolo_detector import YOLODetector
from tracker import get_tracker
from analytics.trajectory_analytics import ZoneAnalytics
from inference.realtime_predictor import RealTimePredictor
from visualization.overlay import SurveillanceOverlay
from utils.config import VIDEO_FPS, OUTPUT_DIR


def run(
    source: str,
    output_path: str | None = None,
    yolo_model: str = "yolov8n.pt",
    device: str = "cpu",
    show: bool = True,
) -> None:
    """
    Main inference loop.

    Args:
        source:       Video path or camera index (as string "0", "1", ...).
        output_path:  Optional path to save annotated output video.
        yolo_model:   YOLO weights file.
        device:       "cpu" or "cuda".
        show:         Display live window.
    """
    # ── Initialise pipeline components ──────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {source}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 1280)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS

    detector  = YOLODetector(yolo_model, device=device)
    tracker   = get_tracker()
    analytics = ZoneAnalytics()
    predictor = RealTimePredictor()
    overlay_r = SurveillanceOverlay(w, h)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        print(f"[Inference] Writing output → {output_path}")

    # ── State variables ──────────────────────────────────────────────────────────
    frame_idx  = 0
    second_idx = 0
    risk_label = "SAFE"
    risk_probs = None
    confidence = 1.0
    fps_timer  = time.time()

    print(f"[Inference] Running on {source}  {w}×{h} @ {fps:.0f} FPS")
    print("[Inference] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Per-frame detection + tracking ────────────────────────────────────────
        detections = detector.detect(frame)
        tracks     = tracker.update(detections, frame, frame_idx)
        states     = tracker.get_all_states()
        analytics.ingest_frame(tracks, states, frame_idx)

        # ── Once-per-second: extract features → GRU prediction ───────────────────
        if frame_idx > 0 and frame_idx % VIDEO_FPS == 0:
            feat = analytics.flush_window()
            risk_label, risk_probs, confidence = predictor.predict(feat)
            second_idx += 1

            print(f"[t={second_idx:4d}s]  "
                  f"people={feat['people_count']:5.1f}  "
                  f"density={feat['density']:.3f}  "
                  f"stagnation={feat['stagnation_ratio']:.2f}  "
                  f"→ {risk_label} ({confidence:.0%})")

        # ── Render overlays ───────────────────────────────────────────────────────
        annotated = overlay_r.render(
            frame, tracks, states,
            risk_label=risk_label,
            risk_probs=risk_probs,
            confidence=confidence,
            second_idx=second_idx,
            is_ready=predictor.is_ready(),
        )

        # FPS counter
        elapsed = time.time() - fps_timer
        if elapsed > 0:
            inf_fps = 1.0 / elapsed
            cv2.putText(annotated, f"FPS: {inf_fps:.1f}", (w - 110, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        fps_timer = time.time()

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("Crowd Risk Monitor — FOOD_COURT_A", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[Inference] User quit.")
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[Inference] Processed {frame_idx} frames ({second_idx} seconds).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Crowd Risk Inference")
    parser.add_argument("--source", type=str, default="0",  help="Video path or camera index")
    parser.add_argument("--output", type=str, default=None, help="Output video path (optional)")
    parser.add_argument("--model",  type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    run(
        source=args.source,
        output_path=args.output,
        yolo_model=args.model,
        device=args.device,
        show=not args.no_show,
    )
