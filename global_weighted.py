# ensemble/global_weighted.py
import numpy as np

def weighted_average_logits(model_logits: dict, weights: dict):
    total = None
    wsum = 0.0
    for m, L in model_logits.items():
        w = float(weights[m])
        total = L * w if total is None else total + L * w
        wsum += w
    return total / max(wsum, 1e-12)

def logits_to_pred(logits: np.ndarray):
    return logits.argmax(axis=1).astype(int)
