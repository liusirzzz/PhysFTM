"""Projected RKHS basis solve routines."""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from physftm.rkhs.basis import RKHSFunctionalTucker2D
from physftm.rkhs.core_solvers import _batched_regularized_solve

Tensor = torch.Tensor


@torch.no_grad()
def ridge_projected_solve_rkhs_mode_coef_2d_from_sparse_mask(
    model: "RKHSFunctionalTucker2D",
    core_batch: Tensor,
    y: Tensor,
    mask: Tensor,
    coord_uniques: Tuple[Tensor, Tensor],
    mode: str,
    lambda_rkhs: float = 0.0001,
    value_ridge: float = 1e-06,
    target_clip: float = 0.95,
    precomputed_basis: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tensor:
    """2D RKHS basis solve for the paper optimization ablation via row-wise target solve + kernel projection.

    This supports both:
      - scalar 2D fields: y=(B,T,H,W,1), core=(B,T,Ru,Rv)
      - shared-basis multichannel 2D fields: y=(B,T,H,W,C), core=(B,T,C,Ru,Rv)

    It first solves for the target basis values on the observed grid rows/columns
    with the opposite mode and core fixed, then projects those targets back to
    RKHS coefficients using kernel ridge regression.

    Important:
      - The current RKHS mode factor is linear in the kernel features: `Kx @ coef`.
    """
    if y.shape != mask.shape:
        raise ValueError(
            f"y and mask must have same shape. got y={tuple(y.shape)} mask={tuple(mask.shape)}"
        )
    if core_batch.dim() not in (4, 5):
        raise ValueError(
            f"2D RKHS basis solve expects scalar core (B,T,Ru,Rv) or shared multichannel core (B,T,C,Ru,Rv), got shape={tuple(core_batch.shape)}"
        )
    is_scalar_2d = y.shape[-1] == 1 and core_batch.dim() == 4
    is_shared_multichannel_2d = y.shape[-1] > 1 and core_batch.dim() == 5
    if not (is_scalar_2d or is_shared_multichannel_2d):
        raise ValueError(
            f"2D RKHS basis solve supports either scalar 2D y=(B,T,H,W,1), core=(B,T,Ru,Rv) or shared-basis multichannel 2D y=(B,T,H,W,C), core=(B,T,C,Ru,Rv); got y={tuple(y.shape)} core={tuple(core_batch.shape)}"
        )
    mode = str(mode).lower()
    if mode not in ("u", "v"):
        raise ValueError(f"mode must be 'u' or 'v', got {mode}")
    if precomputed_basis is not None:
        U_full, V_full = precomputed_basis
    else:
        u_all, v_all = coord_uniques
        model.mode = "training"
        U_full, V_full = model(input_ind_train=(u_all, v_all))
    orig_dtype = core_batch.dtype
    solve_dtype = (
        torch.float32
        if core_batch.dtype in (torch.float16, torch.bfloat16)
        else core_batch.dtype
    )
    y2 = y.to(solve_dtype)
    m2 = mask.to(solve_dtype)
    core = core_batch.to(solve_dtype)
    U_full = U_full.to(solve_dtype)
    V_full = V_full.to(solve_dtype)
    eye_eps = 1e-06
    lambda_rkhs = float(max(lambda_rkhs, 0.0))
    value_ridge = float(max(value_ridge, 0.0))
    target_clip = float(target_clip)
    if mode == "u":
        basis_current = U_full
        other_basis = V_full
        if is_scalar_2d:
            contracted = torch.einsum("js,btrs->btjr", other_basis, core)
            normal = torch.einsum("btijc,btjr,btjs->irs", m2, contracted, contracted)
            rhs = torch.einsum("btijc,btijc,btjr->ir", m2, y2, contracted)
            obs_mass = m2.sum(dim=(0, 1, 3, 4))
        else:
            contracted = torch.einsum("js,btcrs->btcjr", other_basis, core)
            normal = torch.einsum("btijc,btcjr,btcjs->irs", m2, contracted, contracted)
            rhs = torch.einsum("btijc,btijc,btcjr->ir", m2, y2, contracted)
            obs_mass = m2.sum(dim=(0, 1, 3, 4))
        coord_all = coord_uniques[0]
        mode_factor = model.mode_u
    else:
        basis_current = V_full
        if is_scalar_2d:
            contracted = torch.einsum("ir,btrs->btis", U_full, core)
            normal = torch.einsum("btijc,btir,btis->jrs", m2, contracted, contracted)
            rhs = torch.einsum("btijc,btijc,btir->jr", m2, y2, contracted)
            obs_mass = m2.sum(dim=(0, 1, 2, 4))
        else:
            contracted = torch.einsum("ir,btcrs->btcis", U_full, core)
            normal = torch.einsum("btijc,btcir,btcis->jrs", m2, contracted, contracted)
            rhs = torch.einsum("btijc,btijc,btcir->jr", m2, y2, contracted)
            obs_mass = m2.sum(dim=(0, 1, 2, 4))
        coord_all = coord_uniques[1]
        mode_factor = model.mode_v
    solved_values = _batched_regularized_solve(
        normal=normal, rhs=rhs, base_ridge=value_ridge, eye_eps=eye_eps, max_attempts=6
    )
    target_values = basis_current.clone()
    observed_mask = obs_mass > 0
    target_values[observed_mask] = solved_values[observed_mask]
    kernel_xz = mode_factor.kernel(coord_all).to(solve_dtype)
    kernel_zz = mode_factor.kernel.gram().to(solve_dtype)
    if target_clip > 0.0:
        projected_target = target_values.clamp(min=-target_clip, max=target_clip)
    else:
        projected_target = target_values
    lhs = kernel_xz.transpose(0, 1) @ kernel_xz
    lhs = lhs + lambda_rkhs * kernel_zz
    lhs = lhs + eye_eps * torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype)
    rhs_coef = kernel_xz.transpose(0, 1) @ projected_target
    solved_coef = torch.linalg.solve(lhs, rhs_coef)
    return solved_coef.to(orig_dtype)
