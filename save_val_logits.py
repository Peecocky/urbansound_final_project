import numpy as np
import torch
from pathlib import Path
from config.base import LOGIT_DIR, NUM_CLASSES

@torch.no_grad()
def save_val_logits(model, val_loader, model_name, base_seed, fold_id, device):
    model.eval()
    logits_list = []
    labels_list = []
    ids_list = []

    for x, y, sid in val_loader:
        x = x.to(device, non_blocking=True)
        out = model(x)  # (B,10)
        logits_list.append(out.detach().cpu().numpy())
        labels_list.append(y.numpy())
        ids_list.append(sid.numpy())

    logits = np.concatenate(logits_list, axis=0).astype(np.float32)
    labels = np.concatenate(labels_list, axis=0).astype(np.int64)
    ids_ = np.concatenate(ids_list, axis=0).astype(np.int64)

    out_dir = Path(LOGIT_DIR) / "val" / model_name / f"seed{base_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fold{fold_id}.npz"

    np.savez_compressed(out_path, logits=logits, labels=labels, ids=ids_)
    print(f"[OOF] saved {out_path} | logits={logits.shape}", flush=True)
