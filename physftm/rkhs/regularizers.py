"""Regularizers for RKHS Tucker training."""

from __future__ import annotations
from typing import Optional
import torch

Tensor = torch.Tensor


def temporal_gp_penalty(
    core_batch: Tensor,
    time_points: Tensor,
    lengthscale: Optional[float] = None,
    jitter: float = 1e-06,
    chol: Optional[Tensor] = None,
) -> Tensor:
    """Temporal GP quadratic penalty: sum_d g_d^T K^{-1} g_d / D.

    Uses Cholesky + cholesky_solve by default (more stable and faster than explicit inverse).

    Args:
      chol: Optional precomputed Cholesky factor of Kt (lower-triangular), shape (T,T).
            If provided, time_points/lengthscale/jitter are ignored and chol is used directly.
    """
    G = core_batch.reshape(core_batch.shape[0], core_batch.shape[1], -1)
    if chol is None:
        t = torch.as_tensor(
            time_points, dtype=core_batch.dtype, device=core_batch.device
        ).flatten()
        if lengthscale is None:
            span = float(t.max() - t.min() + 1e-06)
            lengthscale = max(span / max(t.numel() - 1, 1), 0.01)
        sqdist = (t[:, None] - t[None, :]).pow(2)
        Kt = torch.exp(-0.5 * sqdist / lengthscale**2)
        Kt = Kt + jitter * torch.eye(t.numel(), device=t.device, dtype=t.dtype)
        chol = torch.linalg.cholesky(Kt)
    return temporal_gp_penalty_from_chol(G, chol)


def temporal_gp_penalty_from_chol(G_or_core_batch: Tensor, chol: Tensor) -> Tensor:
    """Compute temporal GP penalty from a Cholesky factor.

    Inputs:
      - G_or_core_batch: (B,T,*) core batch OR already-flattened G with shape (B,T,D)
      - chol: Cholesky factor (lower) of Kt, shape (T,T)

    Returns:
      Scalar penalty: (G * Kt^{-1} G).sum() / D
    """
    if G_or_core_batch.dim() == 3:
        G = G_or_core_batch
    else:
        G = G_or_core_batch.reshape(
            G_or_core_batch.shape[0], G_or_core_batch.shape[1], -1
        )
    B, T, D = G.shape
    G2 = G.permute(1, 0, 2).reshape(T, B * D)
    X2 = torch.cholesky_solve(G2, chol)
    X = X2.reshape(T, B, D).permute(1, 0, 2)
    return (G * X).sum() / D
