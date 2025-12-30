import random
import numpy as np
import torch
import copy

# =========================
# Seed
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# EMA
# =========================

class ModelEMA:
    """
    Exponential Moving Average for model weights
    """

    def __init__(self, model, decay=0.995, device="cuda"):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)

        self.ema_model.to(device)
        self.ema_model.eval()

        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema_model.state_dict()

        for k in esd.keys():
            if k in msd:
                if esd[k].dtype.is_floating_point:
                    esd[k].mul_(self.decay).add_(
                        msd[k].detach(), alpha=1.0 - self.decay
                    )
                else:
                    esd[k].copy_(msd[k])
