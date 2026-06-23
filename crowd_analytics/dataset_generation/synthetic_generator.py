"""
synthetic_generator.py — Generates realistic synthetic crowd feature CSV datasets.

Why synthetic data?
  Training a real-world surveillance model requires labelled video footage,
  which is expensive and privacy-sensitive. Synthetic generation lets us:
    1. Bootstrap the GRU with realistic temporal patterns.
    2. Control class balance precisely.
    3. Iterate quickly without hardware dependencies.

Simulation approach:
  We model three crowd regimes with smooth temporal transitions:
    SAFE    → low density, flowing movement, low stagnation
    WARNING → building density, slower movement, directional conflicts emerging
    HIGH    → dense, chaotic, high stagnation — congestion peak

  Gaussian noise + temporal auto-correlation (AR(1)) adds realistic variability.
  Smooth transitions between regimes mirror real crowd buildup patterns.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from analytics.risk_labeler import label_risk
from utils.config import FEATURE_COLUMNS, LABEL_COLUMN, DATASET_CSV


# ── Regime parameter table ────────────────────────────────────────────────────
# Each row: (mean, std) for the features in FEATURE_COLUMNS order
_REGIME_PARAMS = {
    #                  cnt   dens   spd   stgn  dwl   flwc  dent  accv  ztrn  inf  outf
    "SAFE":    dict(
        people_count        = (15, 5),
        density             = (0.10, 0.03),
        avg_speed_mps       = (1.20, 0.20),
        stagnation_ratio    = (0.08, 0.05),
        avg_dwell_time_sec  = (30,  10),
        flow_conflict_ratio = (0.10, 0.05),
        directional_entropy = (0.35, 0.10),
        acceleration_variance=(0.005,0.003),
        zone_transitions    = (4,   2),
        inflow              = (2,   1),
        outflow             = (2,   1),
    ),
    "WARNING": dict(
        people_count        = (45, 10),
        density             = (0.35, 0.06),
        avg_speed_mps       = (0.80, 0.15),
        stagnation_ratio    = (0.38, 0.08),
        avg_dwell_time_sec  = (70,  20),
        flow_conflict_ratio = (0.35, 0.10),
        directional_entropy = (0.60, 0.10),
        acceleration_variance=(0.020,0.008),
        zone_transitions    = (8,   3),
        inflow              = (4,   2),
        outflow             = (3,   2),
    ),
    "HIGH":    dict(
        people_count        = (90, 12),
        density             = (0.68, 0.08),
        avg_speed_mps       = (0.30, 0.12),
        stagnation_ratio    = (0.72, 0.08),
        avg_dwell_time_sec  = (130, 25),
        flow_conflict_ratio = (0.62, 0.10),
        directional_entropy = (0.85, 0.07),
        acceleration_variance=(0.050,0.015),
        zone_transitions    = (3,   2),
        inflow              = (2,   1),
        outflow             = (1,   1),
    ),
}

_AR_COEFF = 0.75   # Auto-regression coefficient — controls temporal smoothness


def _sample_regime(regime: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw n samples from the given regime using AR(1) temporal correlation.
    Returns shape (n, len(FEATURE_COLUMNS)).
    """
    params = _REGIME_PARAMS[regime]
    cols   = FEATURE_COLUMNS
    result = np.zeros((n, len(cols)))

    for ci, col in enumerate(cols):
        mu, sigma = params[col]
        # Initialise from stationary distribution
        x = rng.normal(mu, sigma)
        for t in range(n):
            noise = rng.normal(0, sigma * (1 - _AR_COEFF))
            x = _AR_COEFF * x + (1 - _AR_COEFF) * mu + noise
            result[t, ci] = max(x, 0.0)   # clip negatives

    return result


def _interpolate_regimes(
    r1_data: np.ndarray,
    r2_data: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Linear interpolation between two regime segments."""
    out = np.zeros((n_steps, r1_data.shape[1]))
    for t in range(n_steps):
        alpha = t / max(n_steps - 1, 1)
        out[t] = (1 - alpha) * r1_data[t % len(r1_data)] + alpha * r2_data[t % len(r2_data)]
    return out


# ─── Public API ─────────────────────────────────────────────────────────────────

def generate_dataset(
    total_seconds: int = 3600,
    seed: int = 42,
    output_path: Optional[Path] = None,
    target_distribution: Tuple[float, float, float] = (0.45, 0.30, 0.25),
) -> pd.DataFrame:
    """
    Generate a synthetic crowd feature dataset with realistic temporal patterns.

    Args:
        total_seconds:        Number of 1-second feature rows to generate.
        seed:                 Random seed for reproducibility.
        output_path:          If provided, saves CSV to this path.
        target_distribution:  Desired (SAFE, WARNING, HIGH) class fractions.

    Returns:
        DataFrame with columns = FEATURE_COLUMNS + [LABEL_COLUMN, 'timestamp']
    """
    rng = np.random.default_rng(seed)

    safe_n    = int(total_seconds * target_distribution[0])
    warning_n = int(total_seconds * target_distribution[1])
    high_n    = total_seconds - safe_n - warning_n

    print(f"[SyntheticGen] Generating {total_seconds} timesteps  "
          f"SAFE={safe_n}  WARNING={warning_n}  HIGH={high_n}")

    # Generate base segments
    safe_data    = _sample_regime("SAFE",    safe_n,    rng)
    warning_data = _sample_regime("WARNING", warning_n, rng)
    high_data    = _sample_regime("HIGH",    high_n,    rng)

    TRANS = 20   # 20-second transition period between regimes

    segments: List[np.ndarray] = []

    # Build a realistic day scenario: morning ramp, lunch peak, afternoon moderate
    # Segment order: SAFE → WARNING → HIGH → WARNING → SAFE → WARNING → HIGH → SAFE
    schedule = [
        ("SAFE",    safe_n // 3),
        ("WARNING", warning_n // 3),
        ("HIGH",    high_n // 2),
        ("WARNING", warning_n // 3),
        ("SAFE",    safe_n // 3),
        ("WARNING", warning_n // 3),
        ("HIGH",    high_n // 2),
        ("SAFE",    safe_n - 2*(safe_n // 3)),
    ]

    raw_segs = []
    for regime, n in schedule:
        if n <= 0:
            continue
        seg = _sample_regime(regime, n, rng)
        raw_segs.append((regime, seg))

    # Concatenate with smooth transitions
    all_data  = []
    all_labels= []

    for idx, (regime, seg) in enumerate(raw_segs):
        all_data.append(seg)
        all_labels.extend([regime] * len(seg))

        # Add transition blend into next regime
        if idx < len(raw_segs) - 1:
            next_regime, next_seg = raw_segs[idx + 1]
            n_trans = min(TRANS, len(seg), len(next_seg))
            trans   = _interpolate_regimes(seg[-n_trans:], next_seg[:n_trans], n_trans)
            all_data.append(trans)
            # Label transitions as the target regime
            all_labels.extend([next_regime] * n_trans)

    data_matrix = np.vstack(all_data)
    labels_raw  = all_labels[:len(data_matrix)]

    # Build DataFrame
    df = pd.DataFrame(data_matrix, columns=FEATURE_COLUMNS)

    # Apply rule-based risk labelling as ground truth
    # (This cross-validates that the parameters produce the right class distribution)
    df[LABEL_COLUMN] = [label_risk(row) for row in df[FEATURE_COLUMNS].to_dict("records")]
    df["timestamp"]  = range(len(df))

    # Trim to requested total_seconds
    df = df.iloc[:total_seconds].reset_index(drop=True)

    if output_path is not None:
        df.to_csv(output_path, index=False)
        print(f"[SyntheticGen] Saved {len(df)} rows → {output_path}")
        print(f"[SyntheticGen] Class distribution:\n{df[LABEL_COLUMN].value_counts()}")

    return df


if __name__ == "__main__":
    df = generate_dataset(
        total_seconds=7200,
        output_path=DATASET_CSV,
    )
