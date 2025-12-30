import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.utils import set_seed, ModelEMA
from config import CKPT_DIR, NUM_CLASSES
from data.dataset import UrbanDataset 


def run_single_fold(model_name, builder, exp_cfg, fold, device):
    """
    Train + validate one (model, seed, fold)
    Compatible with CRNN / ResNet via UrbanDataset
    """

    seed = int(exp_cfg["base_seed"])
    set_seed(seed)

    print(f"\n[{model_name}][seed={seed}][fold={fold}] START")

    # =========================
    # Dataset (your implementation)
    # =========================
    train_ds, val_ds = UrbanDataset.build_fold(
        fold_id=fold,
        exp_cfg=exp_cfg
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=exp_cfg["batch_size"],
        shuffle=True,
        num_workers=exp_cfg.get("num_workers", 2),
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=exp_cfg["batch_size"],
        shuffle=False,
        num_workers=exp_cfg.get("num_workers", 2),
        pin_memory=(device == "cuda"),
    )

    # =========================
    # Model
    # =========================
    model = builder(NUM_CLASSES).to(device)
    ema = ModelEMA(
        model,
        decay=exp_cfg.get("ema_decay", 0.995),
        device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_cfg["lr"],
        weight_decay=exp_cfg.get("weight_decay", 1e-4)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=exp_cfg["epochs"]
    )

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=exp_cfg.get("label_smoothing", 0.0)
    )

    # =========================
    # Checkpoint
    # =========================
    ckpt_dir = CKPT_DIR / model_name / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fold{fold}.pth"

    best_acc = 0.0
    start_epoch = 1

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        ema.ema_model.load_state_dict(state["model"])
        best_acc = state.get("best_acc", 0.0)
        start_epoch = state.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # =========================
    # Train
    # =========================
    for epoch in range(start_epoch, exp_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y, _ in tqdm(train_loader, leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

        scheduler.step()

        # =========================
        # Validation
        # =========================
        eval_model = model if epoch < 5 else ema.ema_model
        eval_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = eval_model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"[epoch {epoch}/{exp_cfg['epochs']}] val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "model": ema.ema_model.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                },
                ckpt_path,
            )

    print(f"[{model_name}][seed={seed}][fold={fold}] DONE best_acc={best_acc:.4f}")
