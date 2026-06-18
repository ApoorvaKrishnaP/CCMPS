import math
import numpy as np

def dir_entropy(dirs: list) -> float:
    """
    Computes the directional entropy of pedestrian movement vectors.
    Returns a float between 0.0 (orderly, single-direction) and 1.0 (highly chaotic).
    """
    if not dirs: 
        return 0.0
    
    # 1. Distribute angles (0-360) into 8 directional bins (45 degrees each)
    h, _ = np.histogram(dirs, bins=8, range=(0, 360))
    
    # 2. Add a tiny epsilon (1e-9) to prevent log(0) runtime mathematical errors
    p = (h.astype(float) + 1e-9)
    p /= p.sum()
    
    # 3. Calculate Shannon Entropy normalized to a max value of 1.0
    return float(min(-np.sum(p * np.log(p)) / math.log(8), 1.0))


def flow_conflict(dirs: list) -> float:
    """
    Evaluates the geometric intersection profiles of pedestrian trajectories.
    Computes the ratio of opposing directional vectors (>120 degrees of variance).
    """
    if len(dirs) < 2: 
        return 0.0
    
    d = np.array(dirs)
    conflict_count = 0
    total_pairs = 0
    
    # Pairwise comparison loop to check intersection paths between all tracked entities
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            # Calculate the acute angular difference considering the 360-degree boundary wrap
            diff = min(abs(d[i] - d[j]), 360 - abs(d[i] - d[j]))
            
            # If vectors face significantly against each other (>120°), it's a conflict zone profile
            if diff > 120: 
                conflict_count += 1
            total_pairs += 1
            
    return conflict_count / max(total_pairs, 1)