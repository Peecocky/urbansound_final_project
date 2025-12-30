import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import torch
from torch.utils.data import Dataset

from config.base import TRAIN_CSV, AUDIO_DIR

# -------------------------
# helpers
# -------------------------
def _load_wave(path: Path, sr: int, duration: float):
    y, _ = librosa.load(str(path), sr=sr)
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    y = librosa.util.normalize(y)
    return y.astype(np.float32)

def _norm01(m):
    mn, mx = m.min(), m.max()
    return ((m - mn) / (mx - mn + 1e-8)).astype(np.float32)

# -------------------------
# ResNet feature: mel + delta + delta2
# -------------------------
def _resnet_mel(path: Path, cfg: dict):
    y = _load_wave(path, cfg["sr"], cfg["duration"])
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg["sr"], n_fft=cfg["n_fft"], hop_length=cfg["hop"], n_mels=cfg["n_mels"]
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = _norm01(mel)
    d1 = librosa.feature.delta(mel)
    d2 = librosa.feature.delta(mel, order=2)
    x = np.stack([mel, d1, d2], axis=0).astype(np.float32)  # (3, F, T)
    return x

def _resnet_random_crop(x, cfg):
    # x: (3,F,T)
    T = x.shape[-1]
    crop_T = int(T * (cfg["crop_duration"] / cfg["duration"]))
    if T <= crop_T:
        return x
    start = np.random.randint(0, T - crop_T)
    return x[..., start:start+crop_T]

def _resnet_pad_to(x, target_T):
    C, F, T = x.shape
    if T == target_T:
        return x
    if T > target_T:
        return x[..., :target_T]
    pad = target_T - T
    return np.pad(x, ((0,0),(0,0),(0,pad)), mode="edge").astype(np.float32)

def _resnet_specaug(x, cfg):
    # x: (3,F,T) do aug on channel 0 only (mel), then re-derive? to match your script, we aug on base mel then recompute deltas is costly.
    # We'll approximate: aug on all channels same mask.
    out = x.copy()
    C, F, T = out.shape
    # time mask
    t = np.random.randint(1, cfg["time_mask"] + 1)
    t0 = np.random.randint(0, max(1, T - t))
    out[:, :, t0:t0+t] = out.mean()
    # freq mask
    f = np.random.randint(1, cfg["freq_mask"] + 1)
    f0 = np.random.randint(0, max(1, F - f))
    out[:, f0:f0+f, :] = out.mean()
    return out

def _resnet_add_noise(x):
    return (x + np.random.randn(*x.shape).astype(np.float32) * 0.01).astype(np.float32)

def _resnet_specmix(x, p=0.5):
    if np.random.rand() > p:
        return x
    out = x.copy()
    C, F, T = out.shape
    t = np.random.randint(max(1, T // 8), max(2, T // 3))
    t0 = np.random.randint(0, max(1, T - t))
    rolled = np.roll(out, shift=np.random.randint(1, T), axis=-1)
    out[..., t0:t0+t] = rolled[..., t0:t0+t]
    return out

# -------------------------
# CRNN feature: multires mel stack (3, F, T)
# -------------------------
def _crnn_multires_mel(path: Path, cfg: dict):
    y = _load_wave(path, cfg["sample_rate"], cfg["duration"])
    mels = []
    for n_fft in cfg["n_fft_list"]:
        mel = librosa.feature.melspectrogram(
            y=y, sr=cfg["sample_rate"],
            n_mels=cfg["n_mels"],
            n_fft=int(n_fft),
            hop_length=cfg["hop_length"],
            fmax=8000
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = _norm01(mel)
        mels.append(mel.astype(np.float32))
    return np.stack(mels, axis=0)  # (3,F,T)

def _crnn_random_time_crop(x, cfg):
    # x: (3,F,T)
    C, F, T = x.shape
    frames_per_sec = cfg["sample_rate"] / cfg["hop_length"]
    crop_T = int(frames_per_sec * cfg["crop_duration"])
    if T <= crop_T:
        return x
    start = np.random.randint(0, T - crop_T)
    return x[..., start:start+crop_T]

def _crnn_center_crop(x, cfg):
    C, F, T = x.shape
    frames_per_sec = cfg["sample_rate"] / cfg["hop_length"]
    crop_T = int(frames_per_sec * cfg["crop_duration"])
    if T <= crop_T:
        return x
    start = (T - crop_T) // 2
    return x[..., start:start+crop_T]

def _crnn_add_noise(x, cfg):
    if np.random.rand() > cfg["noise_prob"]:
        return x
    sigma = np.random.uniform(cfg["noise_std_low"], cfg["noise_std_high"])
    if sigma <= 0:
        return x
    return (x + np.random.randn(*x.shape).astype(np.float32) * sigma).astype(np.float32)

def _crnn_specaug(x, cfg):
    if np.random.rand() > cfg["spec_aug_prob"]:
        return x
    out = x.copy()
    C, F, T = out.shape

    for _ in range(cfg["time_mask_num"]):
        w = np.random.randint(0, cfg["time_mask_max"] + 1)
        if w <= 0 or T - w <= 0:
            continue
        t0 = np.random.randint(0, T - w)
        out[..., t0:t0+w] = out.mean()

    for _ in range(cfg["freq_mask_num"]):
        h = np.random.randint(0, cfg["freq_mask_max"] + 1)
        if h <= 0 or F - h <= 0:
            continue
        f0 = np.random.randint(0, F - h)
        out[:, f0:f0+h, :] = out.mean()

    return out

# -------------------------
# Dataset
# -------------------------
class UrbanDataset(Dataset):
    """
    Returns: x, y, sid
    x shape:
      - resnet: (3, F, T_fixed)
      - crnn:   (3, F, T_crop)
    """
    def __init__(self, df: pd.DataFrame, mode: str, exp_cfg: dict):
        assert mode in ["train", "val"]
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.exp_cfg = exp_cfg
        self.cfg = exp_cfg["raw"]

        # for resnet: compute target_T once (match your script behavior)
        self.target_T = None
        if self.cfg["name"] == "resnet":
            row = self.df.iloc[0]
            path = AUDIO_DIR / f"fold{int(row['fold'])}" / row["slice_file_name"]
            x = _resnet_mel(path, self.cfg)
            self.target_T = x.shape[-1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["slice_file_name"]
        fold = int(row["fold"])
        label = int(row["classID"])
        sid = int(row.get("ID", -1))

        audio_path = AUDIO_DIR / f"fold{fold}" / fname

        if self.cfg["name"] == "resnet":
            x = _resnet_mel(audio_path, self.cfg)  # (3,F,T)

            if self.mode == "train":
                x = _resnet_random_crop(x, self.cfg)
                x = _resnet_pad_to(x, self.target_T)
                x = _resnet_add_noise(x)
                x = _resnet_specmix(x, p=0.5)
                if self.cfg["spec_aug"]:
                    x = _resnet_specaug(x, self.cfg)
            else:
                x = _resnet_pad_to(x, self.target_T)

        elif self.cfg["name"] == "crnn":
            x = _crnn_multires_mel(audio_path, self.cfg)  # (3,F,T)
            if self.mode == "train":
                x = _crnn_random_time_crop(x, self.cfg)
                x = _crnn_add_noise(x, self.cfg)
                x = _crnn_specaug(x, self.cfg)
            else:
                x = _crnn_center_crop(x, self.cfg)
        else:
            raise ValueError(f"Unknown cfg name: {self.cfg['name']}")

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)
        sid = torch.tensor(sid, dtype=torch.long)
        return x, y, sid

    @staticmethod
    def build_fold(fold_id: int, exp_cfg: dict):
        df = pd.read_csv(TRAIN_CSV)
        train_df = df[df["fold"] != fold_id].reset_index(drop=True)
        val_df   = df[df["fold"] == fold_id].reset_index(drop=True)
        train_ds = UrbanDataset(train_df, mode="train", exp_cfg=exp_cfg)
        val_ds   = UrbanDataset(val_df, mode="val", exp_cfg=exp_cfg)
        return train_ds, val_ds
