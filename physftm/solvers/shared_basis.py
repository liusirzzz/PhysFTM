"""Shared-basis multichannel core solve bridge."""

import torch
from physftm.rkhs.core_solvers import (
    ridge_tv_solve_core_2d_from_sparse_mask,
    ridge_tv_solve_core_3d_from_sparse_mask,
)
from physftm.training.wrapper import broadcast_mask_channels


def solve_shared_basis_multichannel_core_2d_from_sparse_mask(
    *,
    basis,
    y,
    mask,
    coord_uniques,
    rank,
    lambda_ridge,
    beta_tv,
    diag_penalty=None,
    max_cg_iter=30,
    cg_tol=0.0001,
    initial_core=None,
    precomputed_basis=None,
):
    if y.dim() != 5:
        raise ValueError(
            f"shared-basis multichannel core solve expects y dim=5, got shape={tuple(y.shape)}"
        )
    mask = broadcast_mask_channels(mask, y)
    num_channels = int(y.shape[-1])
    solved_per_channel = []
    for channel_idx in range(num_channels):
        init_c = None if initial_core is None else initial_core[:, :, channel_idx, :, :]
        solved_c = ridge_tv_solve_core_2d_from_sparse_mask(
            basis=basis,
            y=y[..., channel_idx : channel_idx + 1],
            mask=mask[..., channel_idx : channel_idx + 1],
            coord_uniques=coord_uniques,
            rank=rank,
            lambda_ridge=lambda_ridge,
            beta_tv=beta_tv,
            diag_penalty=diag_penalty,
            max_cg_iter=max_cg_iter,
            cg_tol=cg_tol,
            initial_core=init_c,
            precomputed_basis=precomputed_basis,
        )
        solved_per_channel.append(solved_c.unsqueeze(2))
    return torch.cat(solved_per_channel, dim=2)


def solve_shared_basis_multichannel_core_3d_from_sparse_mask(
    *,
    basis,
    y,
    mask,
    coord_uniques,
    rank,
    lambda_ridge,
    beta_tv,
    max_cg_iter=30,
    cg_tol=0.0001,
    initial_core=None,
    precomputed_basis=None,
):
    if y.dim() != 6:
        raise ValueError(
            f"shared-basis 3D solve expects y dim=6, got shape={tuple(y.shape)}"
        )
    mask = broadcast_mask_channels(mask, y)
    solved = []
    for channel_idx in range(int(y.shape[-1])):
        init_c = (
            None if initial_core is None else initial_core[:, :, channel_idx, :, :, :]
        )
        solved_c = ridge_tv_solve_core_3d_from_sparse_mask(
            basis=basis,
            y=y[..., channel_idx : channel_idx + 1],
            mask=mask[..., channel_idx : channel_idx + 1],
            coord_uniques=coord_uniques,
            rank=rank,
            lambda_ridge=lambda_ridge,
            beta_tv=beta_tv,
            max_cg_iter=max_cg_iter,
            cg_tol=cg_tol,
            initial_core=init_c,
            precomputed_basis=precomputed_basis,
        )
        solved.append(solved_c.unsqueeze(2))
    return torch.cat(solved, dim=2)
