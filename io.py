import numpy as np
import pandas as pd
from pathlib import Path


# =========================
# Directories (self-contained)
# =========================
ARTIFACT_DIR = Path("artifacts")
LOGIT_DIR = ARTIFACT_DIR / "test_logits"
SUB_DIR = ARTIFACT_DIR / "submissions"

LOGIT_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Save / Load logits
# =========================
def save_logits(model_name: str, seed: int, logits: np.ndarray):
    """
    Save test logits for later ensemble / reuse.
    """
    path = LOGIT_DIR / f"{model_name}_seed{seed}.npy"
    np.save(path, logits)
    return path


def load_logits(model_name: str, seed: int):
    path = LOGIT_DIR / f"{model_name}_seed{seed}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


# =========================
# Save submission
# =========================
def save_submission(ids, preds, out_path: str = "submission.csv"):
    """
    ids   : list[int]
    preds : array-like (N,)
    """
    out_path = Path(out_path)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".csv")

    # default: put under artifacts/submissions
    if not out_path.is_absolute():
        out_path = SUB_DIR / out_path

    df = pd.DataFrame({
        "ID": ids,
        "TARGET": preds
    })
    df.to_csv(out_path, index=False)
    return out_path
