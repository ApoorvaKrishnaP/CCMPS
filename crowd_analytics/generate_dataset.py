"""
generate_dataset.py — Generate synthetic training data and optionally run GRU training.

Usage:
    # Generate 2-hour synthetic dataset
    python generate_dataset.py --seconds 7200

    # Generate + immediately train
    python generate_dataset.py --seconds 7200 --train
"""
import argparse
from pathlib import Path
from dataset_generation.synthetic_generator import generate_dataset
from utils.config import DATASET_CSV

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=7200, help="Timesteps to generate")
    parser.add_argument("--output",  type=str, default=str(DATASET_CSV))
    parser.add_argument("--train",   action="store_true", help="Also run GRU training")
    args = parser.parse_args()

    df = generate_dataset(total_seconds=args.seconds, output_path=Path(args.output))
    print(f"\nGenerated {len(df)} rows.")
    print(df.describe().round(3))

    if args.train:
        from train_gru import train
        train(Path(args.output))
