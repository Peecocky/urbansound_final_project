import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from collections import defaultdict

OOF_DIR = Path("artifacts/oof")

def pick_best_seed(model_name: str):
    """
    返回：
      best_seed, {seed: oof_acc}
    """
    files = list(OOF_DIR.glob(f"{model_name}_seed*_fold*.npz"))
    assert len(files) > 0, f"No OOF files for {model_name}"

    data = defaultdict(list)

    for f in files:
        name = f.stem
        # e.g. crnn_seed350234_fold1
        parts = name.split("_")
        seed = int(parts[1].replace("seed", ""))

        npz = np.load(f)
        logits = npz["logits"]
        labels = npz["labels"]

        preds = logits.argmax(axis=1)
        acc = accuracy_score(labels, preds)

        data[seed].append((acc, len(labels)))

    seed_acc = {}
    for seed, items in data.items():
        total_correct = 0
        total = 0
        for acc, n in items:
            total_correct += acc * n
            total += n
        seed_acc[seed] = total_correct / total

    best_seed = max(seed_acc, key=seed_acc.get)
    return best_seed, seed_acc


if __name__ == "__main__":
    for model in ["crnn", "resnet18", "resnet34", "resnet50"]:
        try:
            best_seed, scores = pick_best_seed(model)
            print(f"[{model}] best_seed={best_seed}")
            for s, a in scores.items():
                print(f"  seed {s}: OOF acc={a:.4f}")
        except AssertionError:
            pass
