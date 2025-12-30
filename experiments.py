import itertools
import random
from copy import deepcopy

# -------------------------
# base search space
# -------------------------

BASE_SEEDS = {
    "crnn":    [350234, 350235, 350236, 350237],
    "resnet":  [103124, 103125, 103126, 103127],
}

LR_SPACE = {
    "crnn":    [3e-4, 2e-4],
    "resnet":  [3e-4, 2.5e-4],
}

LABEL_SMOOTH = {
    "crnn": [0.1],
    "resnet": [0.05],
}

EPOCHS = {
    "crnn": 40,
    "resnet": 50,
}

BATCH_SIZE = 32


# -------------------------
# raw model config (you already use this in dataset)
# -------------------------

RAW_CFG = {
    "crnn": {
        "name": "crnn",
        "sample_rate": 22050,
        "duration": 4.0,
        "crop_duration": 3.0,
        "n_mels": 128,
        "hop_length": 256,
        "n_fft_list": [400, 1024, 2048],
        "noise_prob": 0.4,
        "noise_std_low": 0.0,
        "noise_std_high": 0.02,
        "spec_aug_prob": 0.35,
        "time_mask_max": 32,
        "freq_mask_max": 16,
        "time_mask_num": 2,
        "freq_mask_num": 2,
    },
    "resnet": {
        "name": "resnet",
        "sr": 22050,
        "duration": 4.0,
        "crop_duration": 3.0,
        "n_mels": 128,
        "n_fft": 1024,
        "hop": 256,
        "spec_aug": True,
        "time_mask": 18,
        "freq_mask": 8,
    }
}


# -------------------------
# build experiments
# -------------------------

def build_experiments(model_name: str):
    """
    Return list of exp_cfg dicts
    """
    key = "crnn" if model_name == "crnn" else "resnet"

    exps = []
    for seed, lr, ls in itertools.product(
        BASE_SEEDS[key],
        LR_SPACE[key],
        LABEL_SMOOTH[key],
    ):
        exp = {
            "base_seed": seed,
            "lr": lr,
            "epochs": EPOCHS[key],
            "batch_size": BATCH_SIZE,
            "label_smoothing": ls,
            "ema_decay": 0.995,
            "raw": deepcopy(RAW_CFG[key]),
        }
        exps.append(exp)

    random.shuffle(exps)
    return exps
