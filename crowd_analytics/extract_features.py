"""
extract_features.py — Run the full feature extraction pipeline on a video file.

Usage:
    python extract_features.py --video path/to/footage.mp4 --output data/features.csv
"""
import argparse
from feature_extraction.pipeline import FeatureExtractionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--model",  type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    pipeline = FeatureExtractionPipeline(yolo_model=args.model, device=args.device)
    pipeline.run(video_path=args.video, output_csv=args.output)
