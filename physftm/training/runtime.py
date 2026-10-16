"""Small runtime utilities shared by RKHS training code."""

import numpy as np
import torch
from torch import nn


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_requires_grad(module: nn.Module, flag: bool):
    for param in module.parameters():
        param.requires_grad_(flag)


def format_loss_component_log(components, lambda_map, *, use_latest_pde=False):
    if not components:
        return ""
    keys = components[0].keys()
    latest_comp = components[-1]
    avg_comp = {k: np.mean([c[k] for c in components]) for k in keys}
    parts = []
    for k in keys:
        if use_latest_pde and str(k).startswith("pde_point_"):
            value = latest_comp[k]
        else:
            value = avg_comp[k]
        lam = lambda_map.get(k, 0.0)
        parts.append(f"{k}={value:.4e}(x{lam:.1e}={value * lam:.4e})")
    return " | ".join(parts)


def current_model(model, ddp_state):
    return model.module if ddp_state["distributed"] else model


def current_optimizer_lr(optimizer, fallback=0.0) -> float:
    if optimizer is None or not getattr(optimizer, "param_groups", None):
        return float(fallback)
    return float(optimizer.param_groups[0].get("lr", fallback))
