"""RKHS basis factors and Tucker basis modules."""

from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import nn
from physftm.rkhs.kernels import Matern32Kernel1D

Tensor = torch.Tensor


class RKHSModeFactor1D(nn.Module):

    def __init__(
        self,
        inducing_points: Tensor,
        rank: int,
        init_lengthscale: Optional[float] = None,
        kernel_type: str = "matern32",
        coef_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.kernel = Matern32Kernel1D(
            inducing_points=inducing_points,
            init_lengthscale=init_lengthscale,
            kernel_type=kernel_type,
        )
        self.coef = nn.Parameter(self._init_coef(coef_scale, rank))

    def _init_coef(self, coef_scale: float, rank: int) -> Tensor:
        """RKHS-orthogonal initialization for basis coefficients.

        Initialises B such that columns are approximately orthogonal under the
        kernel inner product:  B^T K B ≈ coef_scale^2 * I.

        This avoids different basis functions collapsing to similar shapes.
        """
        m = self.kernel.z.numel()
        K = self.kernel.gram()
        try:
            L = torch.linalg.cholesky(K)
            raw = torch.randn(m, rank, device=K.device, dtype=K.dtype)
            transformed = torch.linalg.solve_triangular(L, raw, upper=False)
            Q, _ = torch.linalg.qr(transformed)
            coef = coef_scale * torch.linalg.solve_triangular(L.T, Q, upper=True)
        except torch.linalg.LinAlgError:
            coef = coef_scale * torch.randn(m, rank, device=K.device, dtype=K.dtype)
        return coef

    def forward(self, x: Tensor) -> Tensor:
        return self.kernel(x) @ self.coef

    def rkhs_penalty(self) -> Tensor:
        Kzz = self.kernel.gram()
        return torch.trace(self.coef.transpose(0, 1) @ Kzz @ self.coef)


class RKHSFunctionalTucker2D(nn.Module):
    """RKHS Tucker basis for 2D fields with spatial factors U/V only."""

    def __init__(
        self,
        ranks: Tuple[int, int],
        inducing_points: Tuple[Tensor, Tensor],
        lengthscales: Optional[Tuple[Optional[float], Optional[float]]] = None,
        kernel_type: str = "matern32",
        coef_scale: float = 0.01,
    ) -> None:
        super().__init__()
        if lengthscales is None:
            lengthscales = (None, None)
        self.r_1, self.r_2 = ranks
        self._mode = "training"
        self.mode_u = RKHSModeFactor1D(
            inducing_points[0],
            self.r_1,
            init_lengthscale=lengthscales[0],
            kernel_type=kernel_type,
            coef_scale=coef_scale,
        )
        self.mode_v = RKHSModeFactor1D(
            inducing_points[1],
            self.r_2,
            init_lengthscale=lengthscales[1],
            kernel_type=kernel_type,
            coef_scale=coef_scale,
        )

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in ["training", "sampling"]:
            raise ValueError("Mode should be 'training' or 'sampling'")
        self._mode = mode

    def kronecker_product_einsum_batched(self, A: Tensor, B: Tensor) -> Tensor:
        return torch.einsum("bac,bkp->bakcp", A, B).reshape(
            A.size(0), A.size(1) * B.size(1), A.size(2) * B.size(2)
        )

    def basis_matrices(
        self, coord_uniques: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        u, v = coord_uniques
        return (self.mode_u(u), self.mode_v(v))

    def forward(self, input_ind_train=None, input_ind_sampl=None):
        if self._mode == "training":
            u, v = input_ind_train
            return (self.mode_u(u), self.mode_v(v))
        if input_ind_sampl is None:
            raise ValueError("input_ind_sampl is required when mode='sampling'")
        if input_ind_sampl.dim() != 2 or int(input_ind_sampl.shape[-1]) != 2:
            raise ValueError(
                "RKHSFunctionalTucker2D sampling mode expects input_ind_sampl shaped as (Nq,2) for (x,y) queries."
            )
        U = self.mode_u(input_ind_sampl[:, :1].reshape(-1)).unsqueeze(1)
        V = self.mode_v(input_ind_sampl[:, 1:2].reshape(-1)).unsqueeze(1)
        return self.kronecker_product_einsum_batched(U, V).squeeze(1)

    def rkhs_penalty(self) -> Tensor:
        return self.mode_u.rkhs_penalty() + self.mode_v.rkhs_penalty()

    def rkhs_penalty_2d(self) -> Tensor:
        return self.rkhs_penalty()


class RKHSFunctionalTucker3D(nn.Module):
    """RKHS Tucker basis for per-time 3D fields with U/V/W factors."""

    spatial_dim = 3

    def __init__(
        self,
        ranks,
        inducing_points,
        lengthscales=None,
        kernel_type="matern32",
        coef_scale=0.01,
    ):
        super().__init__()
        lengthscales = (None, None, None) if lengthscales is None else lengthscales
        self.r_u, self.r_v, self.r_w = tuple((int(x) for x in ranks))
        self._mode = "training"
        factor_coef_scale = max(float(coef_scale), 0.0) ** (1.0 / 3.0)
        factors = []
        for points, rank, ell in zip(
            inducing_points, (self.r_u, self.r_v, self.r_w), lengthscales
        ):
            factors.append(
                RKHSModeFactor1D(
                    points,
                    rank,
                    init_lengthscale=ell,
                    kernel_type=kernel_type,
                    coef_scale=factor_coef_scale,
                )
            )
        self.mode_u, self.mode_v, self.mode_w = factors

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in ("training", "sampling"):
            raise ValueError("Mode should be 'training' or 'sampling'")
        self._mode = mode

    def basis_matrices(self, coord_uniques):
        u, v, w = coord_uniques
        return (self.mode_u(u), self.mode_v(v), self.mode_w(w))

    def forward(self, input_ind_train=None, input_ind_sampl=None):
        if self._mode == "training":
            u, v, w = input_ind_train
            return (self.mode_u(u), self.mode_v(v), self.mode_w(w))
        if (
            input_ind_sampl is None
            or input_ind_sampl.dim() != 2
            or int(input_ind_sampl.shape[-1]) != 3
        ):
            raise ValueError("RKHSFunctionalTucker3D sampling expects (Nq,3) queries.")
        return (
            self.mode_u(input_ind_sampl[:, 0])
            * self.mode_v(input_ind_sampl[:, 1])
            * self.mode_w(input_ind_sampl[:, 2])
        ).reshape(input_ind_sampl.shape[0], -1)

    def rkhs_penalty(self):
        return (
            self.mode_u.rkhs_penalty()
            + self.mode_v.rkhs_penalty()
            + self.mode_w.rkhs_penalty()
        )

    def rkhs_penalty_2d(self):
        return self.rkhs_penalty()
