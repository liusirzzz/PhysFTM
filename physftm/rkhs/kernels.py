"""Kernel modules used by RKHS Tucker basis factors."""

from __future__ import annotations
import math
from typing import Optional
import torch
from torch import nn

Tensor = torch.Tensor


class Matern32Kernel1D(nn.Module):
    """One-dimensional Matérn-3/2 kernel with fixed inducing points."""

    def __init__(
        self,
        inducing_points: Tensor,
        init_lengthscale: Optional[float] = None,
        init_amplitude: float = 1.0,
        jitter: float = 1e-06,
        kernel_type: str = "matern32",
    ) -> None:
        super().__init__()
        z = torch.as_tensor(inducing_points, dtype=torch.float32).flatten().contiguous()
        if z.numel() == 0:
            raise ValueError("inducing_points must be non-empty")
        self.register_buffer("z", z)
        self.kernel_type = str(kernel_type).lower()
        z_min = torch.tensor(float(z.min()), dtype=torch.float32, device=z.device)
        z_max = torch.tensor(float(z.max()), dtype=torch.float32, device=z.device)
        self.register_buffer("z_min", z_min)
        self.register_buffer("z_max", z_max)
        if init_lengthscale is None:
            span = float(z.max() - z.min() + 1e-06)
            init_lengthscale = max(span / max(z.numel() - 1, 1), 0.01)
        self.log_lengthscale = nn.Parameter(
            torch.tensor(math.log(float(init_lengthscale)), dtype=torch.float32),
            requires_grad=False,
        )
        self.log_amplitude = nn.Parameter(
            torch.tensor(math.log(float(init_amplitude)), dtype=torch.float32),
            requires_grad=False,
        )
        self.jitter = jitter

    def _kernel(self, x: Tensor, z: Tensor) -> Tensor:
        x = x.reshape(-1, 1)
        z = z.reshape(1, -1)
        ell = self.log_lengthscale.exp().clamp_min(1e-05)
        amp = self.log_amplitude.exp().clamp_min(1e-08)
        sqdist = (x - z).pow(2)
        dist = torch.sqrt(sqdist.clamp_min(0.0) + 1e-12)
        scaled = math.sqrt(3.0) * dist / ell
        return amp * (1.0 + scaled) * torch.exp(-scaled)

    def forward(self, x: Tensor) -> Tensor:
        return self._kernel(x.to(self.z), self.z)

    def gram(self) -> Tensor:
        Kzz = self._kernel(self.z, self.z)
        eye = torch.eye(self.z.numel(), device=Kzz.device, dtype=Kzz.dtype)
        return Kzz + self.jitter * eye
