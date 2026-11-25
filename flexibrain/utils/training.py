import numpy as np
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def meta_to_matrix(meta: dict, B: int) -> np.ndarray:
    out = np.empty((B, 4), dtype=np.float32)
    for i in range(B):

        m = meta[i]
        voxel = m.get("voxel", m.get("voxel_size", m.get("spacing")))

        rx = float(voxel[0])
        ry = float(voxel[1])
        rt = float(m.get("rt", voxel[2]))
        tr = float(m["tr"])  

        out[i] = (rx, ry, rt, tr)
    return out

def update_ema(model: nn.Module, momentum: float) -> None:
    """Update target encoder with EMA."""
    if hasattr(model, 'update_target_encoder'):
        model.update_target_encoder(m=momentum)
    elif isinstance(model, DDP) and hasattr(model.module, 'update_target_encoder'):
        model.module.update_target_encoder(m=momentum)


def get_dynamic_momentum(
    epoch: int,
    total_epochs: int,
    base_momentum: float = 0.996,
    final_momentum: float = 0.9999,
    schedule: str = "linear",
) -> float:
    """
    Calculate dynamic momentum for EMA.

    Momentum increases from base_momentum to final_momentum over training
    using either a linear or cosine schedule.
    """
    if total_epochs <= 1:
        progress = 0.0
    else:
        progress = min(1.0, max(0.0, epoch / (total_epochs - 1)))

    if schedule == "linear":
        momentum = base_momentum + (final_momentum - base_momentum) * progress
    elif schedule == "cosine":
        momentum = final_momentum - (final_momentum - base_momentum) * 0.5 * (1 + np.cos(np.pi * progress))
    else:
        raise ValueError(f"Unsupported EMA schedule: {schedule!r}. Expected 'linear' or 'cosine'.")
    return float(momentum)