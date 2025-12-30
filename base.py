from pathlib import Path
import torch

# =========================
# Global seed
# =========================

BASE_SEED = 20050113

# =========================
# Project paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DIR = Path(r"C:\Users\ALIENWARE\Desktop\Kaggle_Data\Kaggle_Data")

AUDIO_DIR = BASE_DIR / "audio"
TRAIN_CSV = BASE_DIR / "metadata" / "kaggle_train.csv"
TEST_CSV = BASE_DIR / "metadata" / "kaggle_test.csv"
SAMPLE_SUB = BASE_DIR / "metadata" / "kaggle_sample_submission.csv"

# =========================
# Checkpoints / outputs
# =========================

CKPT_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Training globals
# =========================

NUM_CLASSES = 10
FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
