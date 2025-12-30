import numpy as np
import json
from pathlib import Path
from config import FOLDS, METRIC_DIR
from inference.io import val_logit_path

def compute_oof(model_name: str, seed: int):
    all_logits, all_labels = [], []
    for f in FOLDS:
        p = val_logit_path(model_name, seed, f)
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        all_logits.append(d["logits"])
        all_labels.append(d["labels"])
    if not all_logits:
        return None
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    pred = logits.argmax(axis=1)
    acc = float((pred == labels).mean())

    # confusion matrix
    C = int(logits.shape[1])
    cm = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(labels, pred):
        cm[int(t), int(p)] += 1

    # per-class recall / precision / f1 (no sklearn)
    eps = 1e-12
    precision = []
    recall = []
    f1 = []
    for c in range(C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        precision.append(float(prec))
        recall.append(float(rec))
        f1.append(float(2*prec*rec / (prec+rec+eps)))

    out = {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }
    return out

def save_oof_report(model_name: str, seed: int):
    rep = compute_oof(model_name, seed)
    if rep is None:
        return None
    out_dir = METRIC_DIR / "oof" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"seed{seed}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    np.save(out_dir / f"seed{seed}_cm.npy", np.array(rep["confusion_matrix"], dtype=np.int64))
    return rep
