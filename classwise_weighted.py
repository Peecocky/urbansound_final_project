# ensemble/classwise_weighted.py
import numpy as np

def classwise_weighted_logits(model_logits: dict, class_weights: dict):
    models = list(model_logits.keys())
    N, C = next(iter(model_logits.values())).shape
    out = np.zeros((N, C), dtype=np.float32)
    denom = np.zeros((C,), dtype=np.float32)

    for m in models:
        w = np.asarray(class_weights[m], dtype=np.float32)  # (C,)
        out += model_logits[m].astype(np.float32) * w[None, :]
        denom += w
    out /= np.maximum(denom[None, :], 1e-12)
    return out
