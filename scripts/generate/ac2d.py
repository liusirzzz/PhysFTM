"""
生成 AC2D 的 201 帧数据集
IBVP-2 风格：200 GRF 样本 × 4 分辨率，每个 201 帧（t=0~2，步长 0.01）

IC 生成策略：在最高分辨率（n=256）生成 GRF，用 zoom 下采样到粗网格，
保证同一样本不同分辨率的初始条件在物理上一致。

依赖：ac2d_etdrk4.py 需在同目录（提供 _etdrk4_coeffs）
用法：python scripts/generate/ac2d.py
"""

import numpy as np
from scipy.ndimage import zoom
import os

try:
    from ac2d_etdrk4 import _etdrk4_coeffs
except ModuleNotFoundError as exc:
    raise ImportError(
        "AC generation requires the original ac2d_etdrk4.py alongside this script; see README.md. The supplied source archive does not include it."
    ) from exc
N_SAMPLES = 200
RESOLS = [32, 64, 128, 256]
N_HIGH = 256
tspan = np.linspace(0.0, 2.0, 201)
lam = 1.0
eps = 0.01
dt = 0.001
OUT_DIR = "data/raw/ac2d"
os.makedirs(OUT_DIR, exist_ok=True)


def grf_2d(n: int, alpha: float, tau: float, rng: np.random.Generator) -> np.ndarray:
    """生成周期 GRF，谱密度 σ(k²+τ²)^{-α/2}，返回 shape=(n,n)"""
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k)
    k2 = kx**2 + ky**2
    sigma = tau ** (0.5 * (2 * alpha - 2))
    sqrt_eig = sigma * (k2 + tau**2) ** (-alpha / 2.0)
    sqrt_eig[0, 0] = 0.0
    noise = rng.standard_normal((n, n))
    noise_hat = np.fft.fft2(noise)
    return np.fft.ifft2(sqrt_eig * noise_hat).real


def solve_with_ic(
    u0: np.ndarray, tspan: np.ndarray, dt: float, lam: float, eps: float
) -> np.ndarray:
    """用给定初始条件求解 AC 方程，返回 (n, n, T)"""
    n = u0.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k)
    L = lam * eps**2 * -(kx**2 + ky**2)
    e, e2, a32, a43, b2, b3, b4 = _etdrk4_coeffs(L, dt)
    uhat = np.fft.fft2(u0)
    out_steps = np.round(tspan / dt).astype(int)
    step_to_idx = {int(s): i for i, s in enumerate(out_steps)}
    nsteps = int(out_steps[-1])
    nt = len(tspan)
    solution = np.zeros((n, n, nt), dtype=np.float64)
    solution[:, :, 0] = u0.copy()
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
            solution[:, :, idx] = np.fft.ifft2(uhat).real
    return solution


rng_meta = np.random.default_rng(42)
grf_alphas = rng_meta.uniform(2.0, 3.0, N_SAMPLES)
grf_taus = rng_meta.uniform(12.0, 25.0, N_SAMPLES)
for s in range(N_SAMPLES):
    all_done = all(
        (
            os.path.exists(os.path.join(OUT_DIR, f"IBVP2_s{s:03d}_n{n}.npz"))
            for n in RESOLS
        )
    )
    if all_done:
        print(f"[{s + 1:3d}/{N_SAMPLES}] 已存在，跳过", flush=True)
        continue
    print(f"\n[{s + 1:3d}/{N_SAMPLES}] 样本 {s:03d}", flush=True)
    rng_ic = np.random.default_rng(s)
    grf_raw = grf_2d(N_HIGH, grf_alphas[s], grf_taus[s], rng_ic)
    u0_high = grf_raw / grf_raw.std() * 0.3
    for n in RESOLS:
        out_path = os.path.join(OUT_DIR, f"IBVP2_s{s:03d}_n{n}.npz")
        if os.path.exists(out_path):
            print(f"  n={n:3d}  已存在，跳过", flush=True)
            continue
        scale = n / N_HIGH
        u0 = zoom(u0_high, scale, order=3) if n < N_HIGH else u0_high.copy()
        solution = solve_with_ic(u0, tspan, dt, lam, eps)
        x = np.linspace(0.0, 1.0, n, endpoint=False)
        xx, yy = np.meshgrid(x, x)
        np.savez(out_path, solution=solution, tspan=tspan, xx=xx, yy=yy)
        print(f"  n={n:3d}  shape={solution.shape}", flush=True)
print("\n全部完成！", flush=True)
print(f"输出目录：{OUT_DIR}/", flush=True)
print("文件格式：IBVP2_s{000..199}_n{32,64,128,256}.npz", flush=True)
print("每个文件：solution shape=(n, n, 201)，tspan shape=(201,)", flush=True)
