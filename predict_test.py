import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from config.base import TEST_CSV, SAMPLE_SUB
import pandas as pd

def predict_test_for_seed(
    model_name,
    builder,
    seed,
    device,
    DatasetCls,
    batch_size=32,
    temperature=1.0,
):
    ckpt_dir = Path("checkpoints") / model_name / f"seed{seed}"
    ckpts = sorted(ckpt_dir.glob("fold*.pth"))

    df_test = pd.read_csv(TEST_CSV)
    test_ds = DatasetCls(df_test, mode="test")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logits_sum = None
    used = 0

    for ck in ckpts:
        model = builder().to(device)
        model.load_state_dict(torch.load(ck, map_location=device)["model"])
        model.eval()

        all_logits = []
        with torch.no_grad():
            for x, _, _ in test_loader:
                x = x.to(device)
                out = model(x) / temperature
                all_logits.append(out.cpu().numpy())

        fold_logits = np.concatenate(all_logits, axis=0)
        logits_sum = fold_logits if logits_sum is None else logits_sum + fold_logits
        used += 1

    avg_logits = logits_sum / used
    return avg_logits
