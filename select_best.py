import torch
from pathlib import Path
from config import CKPT_DIR, FOLDS

def pick_best_experiments(model_name: str, top_k=2):
    model_dir = CKPT_DIR / model_name
    scores = []

    for seed_dir in model_dir.glob("seed*"):
        seed = int(seed_dir.name.replace("seed", ""))
        accs = []

        for f in FOLDS:
            ckpt = seed_dir / f"fold{f}.pth"
            if not ckpt.exists():
                break
            state = torch.load(ckpt, map_location="cpu")
            accs.append(state["best_acc"])

        if len(accs) == len(FOLDS):
            scores.append({
                "seed": seed,
                "oof_acc": sum(accs) / len(accs)
            })

    scores = sorted(scores, key=lambda x: x["oof_acc"], reverse=True)
    return scores[:top_k]
