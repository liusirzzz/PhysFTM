"""Core solve routines for RKHS Tucker models."""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from physftm.rkhs.basis import RKHSFunctionalTucker2D

Tensor = torch.Tensor


def _cg_solve(
    matvec,
    b: Tensor,
    x0: Optional[Tensor] = None,
    max_iter: int = 50,
    tol: float = 1e-06,
) -> Tensor:
    """Conjugate gradient solver for SPD systems Ax=b.

    Args:
      matvec: callable(v)->Av
      b: (...,N)
      x0: (...,N)
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = (r * r).sum(dim=-1, keepdim=True)
    b_norm = b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    converged = torch.sqrt(rs_old) <= tol * b_norm
    if converged.all():
        return x
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = (p * Ap).sum(dim=-1, keepdim=True)
        valid = ~converged & torch.isfinite(denom) & (denom > 0)
        if not valid.any():
            break
        alpha = torch.zeros_like(rs_old)
        alpha = torch.where(valid, rs_old / denom.clamp_min(1e-12), alpha)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum(dim=-1, keepdim=True)
        converged_new = torch.sqrt(rs_new) <= tol * b_norm
        converged = converged | converged_new | ~valid
        if converged.all():
            break
        beta = torch.zeros_like(rs_old)
        beta = torch.where(valid, rs_new / rs_old.clamp_min(1e-12), beta)
        p = torch.where(valid, r + beta * p, torch.zeros_like(p))
        rs_old = torch.where(valid, rs_new, rs_old)
    return x


def _batched_regularized_solve(
    normal: Tensor,
    rhs: Tensor,
    base_ridge: float = 1e-06,
    eye_eps: float = 1e-08,
    max_attempts: int = 5,
) -> Tensor:
    """Robust batched linear solve for nearly-singular SPD systems.

    Tries Cholesky with progressively stronger diagonal regularization, then
    falls back to pseudoinverse if some batches remain singular.
    """
    if normal.dim() != 3 or rhs.dim() != 2:
        raise ValueError(
            f"Expected normal=(N,R,R) rhs=(N,R); got {tuple(normal.shape)} and {tuple(rhs.shape)}"
        )
    n_batch, rank_dim, _ = normal.shape
    eye = torch.eye(rank_dim, device=normal.device, dtype=normal.dtype).unsqueeze(0)
    ridge = float(max(base_ridge, 0.0)) + float(max(eye_eps, 0.0))
    stabilized = None
    for attempt in range(max_attempts):
        cur_ridge = ridge * 10.0**attempt
        stabilized = normal + cur_ridge * eye
        chol, info = torch.linalg.cholesky_ex(stabilized)
        if bool((info == 0).all()):
            return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
    if stabilized is None:
        stabilized = normal + ridge * eye
    return (torch.linalg.pinv(stabilized) @ rhs.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def ridge_tv_solve_core_2d_from_sparse_mask(
    basis: "RKHSFunctionalTucker2D",
    y: Tensor,
    mask: Tensor,
    coord_uniques: Tuple[Tensor, Tensor],
    rank: Tuple[int, int, int],
    lambda_ridge: float = 0.0001,
    beta_tv: float = 0.0,
    diag_penalty: Optional[Tensor] = None,
    max_cg_iter: int = 30,
    cg_tol: float = 0.0001,
    initial_core: Optional[Tensor] = None,
    precomputed_basis: Optional[Tuple[Tensor, Tensor]] = None,
) -> Tensor:
    """Fast 2D TV-ridge core solver using sparse contractions.

    The spatial observation set is built as a union over both batch and time.
    This supports temporal frame drops and time-varying spatial masks while
    retaining the sparse contraction path.  Individual ``(b,t)`` entries are
    still weighted by their own mask values during the solve.

    This avoids full-field decode in each CG matvec. Complexity becomes O(Nobs * (Ru+Rv))
    per iteration instead of O(H*W*Ru*Rv).

    Shapes:
      - y, mask: (B,T,H,W,1)
      - rank=(Ru,Rv,1)

    Returns:
      - core: (B,T,Ru,Rv)
    """
    if y.shape != mask.shape:
        raise ValueError(
            f"y and mask must have same shape. got y={tuple(y.shape)} mask={tuple(mask.shape)}"
        )
    if y.shape[-1] != 1:
        raise ValueError("sparse 2D solver expects y.shape[-1]==1")
    Ru, Rv, Rw = rank
    if Rw != 1:
        raise ValueError(f"2D data expects rank[2]==1, got rank={rank}")
    lambda_ridge = float(max(lambda_ridge, 1e-12))
    diag_stabilizer = float(1e-08 * (1.0 + abs(beta_tv)))
    orig_dtype = y.dtype
    solve_dtype = (
        torch.float32 if y.dtype in (torch.float16, torch.bfloat16) else y.dtype
    )
    y = y.to(solve_dtype)
    mask = mask.to(solve_dtype)
    B, T, H, Wd, _ = y.shape
    u_all, v_all = coord_uniques
    if precomputed_basis is not None:
        U, V = precomputed_basis
    else:
        basis.mode = "training"
        U, V = basis(input_ind_train=(u_all, v_all))
    U = U.to(solve_dtype)
    V = V.to(solve_dtype)
    obs_ij = torch.nonzero(mask[..., 0].sum(dim=(0, 1)) > 0, as_tuple=False)
    if obs_ij.numel() == 0:
        return torch.zeros(B, T, Ru, Rv, device=y.device, dtype=orig_dtype)
    ii = obs_ij[:, 0].long()
    jj = obs_ij[:, 1].long()
    U_obs = U[ii]
    V_obs = V[jj]
    y_obs = y[..., 0][:, :, ii, jj]
    w_obs = mask[..., 0][:, :, ii, jj]
    P = Ru * Rv
    diag_penalty_vec = None
    if diag_penalty is not None:
        diag_penalty_vec = torch.as_tensor(
            diag_penalty, dtype=solve_dtype, device=y.device
        ).reshape(1, 1, P)

    def apply_time_laplacian(x: Tensor) -> Tensor:
        if T == 1:
            return torch.zeros_like(x)
        out = torch.zeros_like(x)
        out[:, 0] = x[:, 0] - x[:, 1]
        out[:, -1] = x[:, -1] - x[:, -2]
        if T > 2:
            out[:, 1:-1] = 2 * x[:, 1:-1] - x[:, 0:-2] - x[:, 2:]
        return out

    wy = w_obs * y_obs
    b_mat = torch.einsum("nr, ns, btn -> btrs", U_obs, V_obs, wy)
    b_vec = b_mat.reshape(B, T, P)

    def matvec(x_vec: Tensor) -> Tensor:
        x_mat = x_vec.reshape(B, T, Ru, Rv)
        pred = torch.einsum("nr, ns, btrs -> btn", U_obs, V_obs, x_mat)
        pred = w_obs * pred
        ata = torch.einsum("nr, ns, btn -> btrs", U_obs, V_obs, pred).reshape(B, T, P)
        out = ata + (lambda_ridge + diag_stabilizer) * x_vec
        if diag_penalty_vec is not None:
            out = out + diag_penalty_vec * x_vec
        if beta_tv and beta_tv > 0:
            out = out + beta_tv * apply_time_laplacian(x_vec)
        return out

    x0 = None
    if initial_core is not None:
        x0 = initial_core.reshape(B, T, P).to(dtype=b_vec.dtype, device=b_vec.device)
    sol = _cg_solve(matvec, b_vec, x0=x0, max_iter=max_cg_iter, tol=cg_tol)
    return sol.reshape(B, T, Ru, Rv).to(orig_dtype)


@torch.no_grad()
def ridge_tv_solve_core_3d_from_sparse_mask(
    basis,
    y,
    mask,
    coord_uniques,
    rank,
    lambda_ridge=0.0001,
    beta_tv=0.0,
    max_cg_iter=30,
    cg_tol=0.0001,
    initial_core=None,
    precomputed_basis=None,
):
    """Solve a per-time 3D core from sparse (B,T,D,H,W,1) observations."""
    if y.shape != mask.shape:
        raise ValueError(
            f"y and mask must have same shape. got y={tuple(y.shape)} mask={tuple(mask.shape)}"
        )
    if y.dim() != 6 or int(y.shape[-1]) != 1:
        raise ValueError(f"3D core solve expects y=(B,T,D,H,W,1), got {tuple(y.shape)}")
    Ru, Rv, Rw = (int(x) for x in rank)
    if precomputed_basis is None:
        basis.mode = "training"
        U, V, W = basis(input_ind_train=coord_uniques)
    else:
        U, V, W = precomputed_basis
    solve_dtype = (
        torch.float32 if y.dtype in (torch.float16, torch.bfloat16) else y.dtype
    )
    y = y.to(solve_dtype)
    mask = mask.to(solve_dtype)
    U, V, W = (x.to(solve_dtype) for x in (U, V, W))
    B, T, D, H, X, _ = y.shape
    if (U.shape[0], V.shape[0], W.shape[0]) != (D, H, X):
        raise ValueError("3D basis spatial size mismatch with y")
    obs = torch.nonzero(mask[..., 0].sum(dim=(0, 1)) > 0, as_tuple=False)
    if obs.numel() == 0:
        return torch.zeros(B, T, Ru, Rv, Rw, device=y.device, dtype=y.dtype)
    di, hi, wi = (obs[:, i].long() for i in range(3))
    Uo, Vo, Wo = (U[di], V[hi], W[wi])
    y_obs = y[..., 0][:, :, di, hi, wi]
    m_obs = mask[..., 0][:, :, di, hi, wi]
    P = Ru * Rv * Rw
    b_mat = torch.einsum("nr,ns,nq,btn->btrsq", Uo, Vo, Wo, m_obs * y_obs)
    b_vec = b_mat.reshape(B, T, P)
    lambda_ridge = float(max(lambda_ridge, 1e-12))

    def apply_time_laplacian(x):
        if T <= 1:
            return torch.zeros_like(x)
        out = torch.zeros_like(x)
        out[:, 0] = x[:, 0] - x[:, 1]
        out[:, -1] = x[:, -1] - x[:, -2]
        if T > 2:
            out[:, 1:-1] = 2 * x[:, 1:-1] - x[:, :-2] - x[:, 2:]
        return out

    def matvec(x_vec):
        x_mat = x_vec.reshape(B, T, Ru, Rv, Rw)
        pred = torch.einsum("nr,ns,nq,btrsq->btn", Uo, Vo, Wo, x_mat)
        pred = m_obs * pred
        ata = torch.einsum("nr,ns,nq,btn->btrsq", Uo, Vo, Wo, pred).reshape(B, T, P)
        out = ata + lambda_ridge * x_vec
        if beta_tv > 0:
            out = out + float(beta_tv) * apply_time_laplacian(x_vec)
        return out

    x0 = (
        None
        if initial_core is None
        else initial_core.reshape(B, T, P).to(dtype=b_vec.dtype, device=b_vec.device)
    )
    sol = _cg_solve(matvec, b_vec, x0=x0, max_iter=max_cg_iter, tol=cg_tol)
    return sol.reshape(B, T, Ru, Rv, Rw).to(y.dtype)
