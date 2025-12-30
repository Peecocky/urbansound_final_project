import random
from copy import deepcopy

from config.base import AUDIO_DIR
from data.dataset import UrbanDataset, UrbanSoundMultiResDataset

# ======================================================
# 1. Base (Human-designed) configs
# ======================================================

BASE_CONFIG = {
    "crnn": {
        "dataset_cls": UrbanSoundMultiResDataset,
        "num_workers": 2,
        "epochs": 80,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.10,
        "use_mixup": True,
        "mixup_start_epoch": 3,
        "ema_start_epoch": 5,
        "ema_decay": 0.995,
        "temperature": 1.2,
        "base_seed": 350234,

        # ===== CRNN augmentation base =====
        "time_mask_max": 32,
        "time_mask_num": 2,
        "spec_aug_prob": 0.35,
    },

    "resnet18": {
        "dataset_cls": UrbanDataset,
        "num_workers": 0,
        "epochs": 80,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "use_mixup": True,
        "mixup_start_epoch": 1,
        "ema_start_epoch": 1,
        "ema_decay": 0.995,
        "temperature": 1.5,
        "base_seed": 103124,
    },

    "resnet34": {
        "dataset_cls": UrbanDataset,
        "num_workers": 0,
        "epochs": 30,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "use_mixup": True,
        "mixup_start_epoch": 1,
        "ema_start_epoch": 1,
        "ema_decay": 0.995,
        "temperature": 1.5,
        "base_seed": 203124,
    },

    "resnet50": {
        "dataset_cls": UrbanDataset,
        "num_workers": 0,
        "epochs": 30,
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "use_mixup": True,
        "mixup_start_epoch": 1,
        "ema_start_epoch": 1,
        "ema_decay": 0.995,
        "temperature": 1.5,
        "base_seed": 303124,
    },
}

# ======================================================
# 2. Auto-generate optimized experiments
# ======================================================

def generate_experiments(
    base_cfg: dict,
    n_variants: int,
    seed_offset: int = 1000,
):
    """
    Generate multiple high-quality variants around a base config
    Dataset augmentation is part of the search space (CRNN only).
    """
    exps = []

    for i in range(n_variants):
        cfg = deepcopy(base_cfg)

        # ---- seed ----
        cfg["base_seed"] = base_cfg["base_seed"] + i * seed_offset

        # ---- lr (small log-uniform jitter) ----
        lr_scale = random.choice([0.7, 0.85, 1.0, 1.15, 1.3])
        cfg["lr"] = base_cfg["lr"] * lr_scale

        # ---- label smoothing (very sensitive!) ----
        if cfg["label_smoothing"] > 0:
            cfg["label_smoothing"] = max(
                0.0,
                base_cfg["label_smoothing"] +
                random.choice([-0.02, -0.01, 0.0, 0.01])
            )

        # ---- EMA timing (small safe jitter) ----
        cfg["ema_start_epoch"] = max(
            1,
            base_cfg["ema_start_epoch"] + random.choice([-1, 0, 1])
        )

        # ---- mixup timing ----
        if cfg["use_mixup"]:
            cfg["mixup_start_epoch"] = max(
                1,
                base_cfg["mixup_start_epoch"] + random.choice([-1, 0, 1])
            )

        # ==================================================
        # 🔥 CRNN-only: dataset augmentation search
        # ==================================================
        if cfg.get("dataset_cls") == UrbanSoundMultiResDataset:
            cfg["time_mask_max"] = max(
                16,
                base_cfg["time_mask_max"] + random.choice([-8, 0, 8])
            )

            cfg["time_mask_num"] = max(
                1,
                base_cfg["time_mask_num"] + random.choice([-1, 0, 1])
            )

            cfg["spec_aug_prob"] = min(
                0.6,
                max(
                    0.2,
                    base_cfg["spec_aug_prob"] +
                    random.choice([-0.1, 0.0, 0.1])
                )
            )

        # ---- IO ----
        cfg["audio_dir"] = AUDIO_DIR

        exps.append(cfg)

    return exps


# ======================================================
# 3. Final EXPERIMENTS used by runner
# ======================================================

EXPERIMENTS = {
    # CRNN: model + optimizer + augmentation jointly searched
    "crnn": generate_experiments(BASE_CONFIG["crnn"], n_variants=5),

    # ResNet: stable optimizer/seed search only
    "resnet18": generate_experiments(BASE_CONFIG["resnet18"], n_variants=5),
    "resnet34": generate_experiments(BASE_CONFIG["resnet34"], n_variants=1),
    "resnet50": generate_experiments(BASE_CONFIG["resnet50"], n_variants=1),
}

# Note:
# Experiments show CRNN and ResNet18 dominate ensemble gains.
