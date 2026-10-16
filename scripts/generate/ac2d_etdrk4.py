"""
Allen-Cahn 2D periodic solver using Fourier spectral ETDRK4.

This implementation mirrors Chebfun's SPIN2 + expinteg("etdrk4") logic:
- phi/psi functions via complex-mean contour (see chebfun-master/@expinteg)
- contour construction from chebfun-master/@spinoperator/computeLR.m

Outputs are saved as NPZ files (solution, tspan, xx, yy).
"""

from __future__ import annotations

import os
from math import factorial

import numpy as np


def _phi_fun(l: int, z: np.ndarray) -> np.ndarray:
    if l == 0:
        return np.exp(z)
    f = _phi_fun(l - 1, z)
    return (f - 1.0 / factorial(l - 1)) / z


def _compute_lr(L: np.ndarray, dt: float, m: int) -> np.ndarray:
    if np.isrealobj(L):
        r = np.exp(1j * np.pi * ((np.arange(1, m + 1) - 0.5) / m))
    else:
        r = np.exp(2j * np.pi * ((np.arange(1, m + 1) - 0.5) / m))
    Lflat = L.ravel()[:, None]
    return dt * Lflat + r[None, :]


def _phi_eval(l: int, lr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    vals = np.mean(_phi_fun(l, lr), axis=1)
    return vals.reshape(shape)


def _psi_eval(l: int, c: float, lr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    vals = np.mean((c**l) * _phi_fun(l, c * lr), axis=1)
    return vals.reshape(shape)


def _etdrk4_coeffs(L: np.ndarray, dt: float, m: int = 32):
    lr = _compute_lr(L, dt, m)
    shape = L.shape

    phi2 = np.real(_phi_eval(2, lr, shape))
    phi3 = np.real(_phi_eval(3, lr, shape))
    psi12 = np.real(_psi_eval(1, 0.5, lr, shape))

    e2 = np.exp(0.5 * dt * L)
    e = np.exp(dt * L)

    a32 = dt * psi12
    a43 = dt * (2.0 * psi12)
    b2 = dt * (2.0 * phi2 - 4.0 * phi3)
    b3 = dt * (2.0 * phi2 - 4.0 * phi3)
    b4 = dt * (-phi2 + 4.0 * phi3)

    return e, e2, a32, a43, b2, b3, b4


def solve_ac2d(
    n: int,
    tspan: np.ndarray,
    dt: float,
    lam: float,
    eps: float,
    outdir: str,
    tag: str,
):
    Lx = 1.0
    x = np.linspace(0.0, Lx, n, endpoint=False)
    y = np.linspace(0.0, Lx, n, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    u = np.tanh((0.35 - r) / (2 * eps))

    k = 2 * np.pi * np.fft.fftfreq(n, d=Lx / n)
    kx, ky = np.meshgrid(k, k)
    lap = -(kx**2 + ky**2)
    L = lam * (eps**2) * lap

    e, e2, a32, a43, b2, b3, b4 = _etdrk4_coeffs(L, dt)

    uhat = np.fft.fft2(u)

    nt = len(tspan)
    solution = np.zeros((n, n, nt), dtype=np.float64)
    solution[:, :, 0] = np.real(u)

    out_steps = np.round(tspan / dt).astype(int)
    step_to_idx = {int(s): i for i, s in enumerate(out_steps)}
    nsteps = int(out_steps[-1])

    for step in range(1, nsteps + 1):
        v2_hat = e2 * uhat
        v2 = np.fft.ifft2(v2_hat)
        nv2_hat = np.fft.fft2(-lam * v2**3 + lam * v2)

        v3_hat = e2 * uhat + a32 * nv2_hat
        v3 = np.fft.ifft2(v3_hat)
        nv3_hat = np.fft.fft2(-lam * v3**3 + lam * v3)

        v4_hat = e * uhat + a43 * nv3_hat
        v4 = np.fft.ifft2(v4_hat)
        nv4_hat = np.fft.fft2(-lam * v4**3 + lam * v4)

        uhat = e * uhat + b2 * nv2_hat + b3 * nv3_hat + b4 * nv4_hat

        if step in step_to_idx:
            idx = step_to_idx[step]
            u = np.fft.ifft2(uhat)
            solution[:, :, idx] = np.real(u)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{tag}.npz")
    np.savez(out_path, solution=solution, tspan=tspan, xx=xx, yy=yy)


def main() -> None:
    tspan = np.arange(0.0, 10.0 + 1e-12, 0.1)
    lam = 10.0
    eps = 0.025
    dt = 1e-3

    solve_ac2d(256, tspan, dt, lam, eps, "AC2D1_py", "AC2D1_HR")
    solve_ac2d(64, tspan, dt, lam, eps, "AC2D1_py", "AC2D1_LR")


if __name__ == "__main__":
    main()
