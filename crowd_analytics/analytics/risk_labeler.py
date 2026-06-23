"""
risk_labeler.py — Rule-based risk labelling for CSV dataset generation.

During dataset generation we don't yet have a trained GRU, so we derive
ground-truth labels from deterministic thresholds on key features.

Label logic (applied in priority order):
  HIGH    → density > 0.60 OR stagnation > 0.65 OR (density > 0.45 AND flow_conflict > 0.55)
  WARNING → density > 0.30 OR stagnation > 0.35 OR people_count > 40
  SAFE    → everything else

These thresholds are calibrated to a 150 m² zone with a 120-person capacity.
"""

from utils.config import ZONES, ACTIVE_ZONE


_cfg = ZONES[ACTIVE_ZONE]
_AREA = _cfg["area_m2"]


def label_risk(features: dict) -> str:
    """
    Assign SAFE / WARNING / HIGH based on heuristic feature thresholds.

    Args:
        features: dict matching FEATURE_COLUMNS in config.py

    Returns:
        Risk label string: "SAFE", "WARNING", or "HIGH"
    """
    density      = features.get("density", 0.0)
    stagnation   = features.get("stagnation_ratio", 0.0)
    flow_conf    = features.get("flow_conflict_ratio", 0.0)
    people       = features.get("people_count", 0.0)

    if density > 0.60 or stagnation > 0.65 or (density > 0.45 and flow_conf > 0.55):
        return "HIGH"
    if density > 0.30 or stagnation > 0.35 or people > 40:
        return "WARNING"
    return "SAFE"
