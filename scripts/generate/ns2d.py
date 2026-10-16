"""
Generate a 2D Navier--Stokes dataset in the same physical setting as the
Navier--Stokes example in the Fourier Neural Operator (FNO) paper:

    Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu,
    Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar,
    "Fourier Neural Operator for Parametric Partial Differential Equations",
    arXiv:2010.08895.

The PDE is the incompressible 2D Navier--Stokes equation in vorticity form on
the unit torus:

    ∂_t ω + u · ∇ω = ν Δω + f,          x ∈ (0, 1)^2,  t > 0
    ∇ · u = 0
    -Δψ = ω
    u = (∂_y ψ, -∂_x ψ)

The forcing in this FNO case is

    f(x, y) = 0.1 * (sin(2π(x + y)) + cos(2π(x + y))).

Here:
    - ω is the scalar vorticity,
    - u is the velocity field,
    - ψ is the stream function,
    - ν is the viscosity,
    - f is the fixed forcing term used in the FNO example,
    - the boundary condition is periodic in both spatial directions.

Initial conditions are sampled from the same style of periodic Gaussian random
field used by the original FNO data-generation code.

Numerical method:
    - spatial discretization: pseudospectral / Fourier method,
    - nonlinear term: computed in physical space and transformed back,
    - dealiasing: 2/3 rule,
    - time stepping: Crank--Nicolson update for the viscous term,
    - implementation: the core update follows the original FNO Navier--Stokes
      data-generation script, rewritten in modern torch.fft syntax.

Residual diagnostics:
    - time derivative: forward finite difference,
    - spatial derivatives: periodic central finite differences,
    - Poisson solve for the diagnostic velocity: periodic finite-difference
      Laplacian inverted in Fourier space,
    - because forward difference uses omega(t + dt) - omega(t), the residual is
      defined at all saved times except the final saved time.

How to use this file:
    Run directly for the paper-size configuration, or use --help for overrides.
    The default export has shape (H,W,T,3), channel order (u,v,omega).
    Velocity is computed on the solver grid before spatial subsampling; omega
    comes from the solver. generation_config.json records the effective settings.
    Existing output requires --overwrite to replace.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import shutil
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

NU = 0.0001
SOLVE_NX = 256
SOLVE_NY = 256
SAVE_NX = 128
SAVE_NY = 128
SOLVE_DT = 0.0001
SAVE_DT = 0.1
NUM_STEPS = 200000
NUM_SAMPLES = 800
PARALLEL_CASES = 64
PLOT = False
SEED = 100
SAVE_FIELDS = "uvomega"
GRF_ALPHA = 2.5
GRF_TAU = 7.0
OUTPUT_ROOT = "data/raw/ns2d"


@dataclass(frozen=True)
class RuntimeConfig:
    nu: float
    solve_nx: int
    solve_ny: int
    save_nx: int
    save_ny: int
    solve_dt: float
    save_dt: float
    num_steps: int
    num_samples: int
    parallel_cases: int
    plot: bool
    seed: int
    save_fields: str
    grf_alpha: float
    grf_tau: float
    output_root: Path

    @property
    def solve_size(self) -> int:
        return self.solve_nx

    @property
    def save_stride(self) -> int:
        ratio = self.save_dt / self.solve_dt
        return int(round(ratio))

    @property
    def saved_times(self) -> int:
        return self.num_steps // self.save_stride + 1

    @property
    def parameter_directory_name(self) -> str:
        return f"nu_{format_scientific_tag(self.nu)}_shape_{self.save_nx}x{self.save_ny}x{self.saved_times}_dt_{format_scientific_tag(self.save_dt)}_fields_{self.save_fields}"

    @property
    def parameter_directory(self) -> Path:
        return self.output_root / self.parameter_directory_name


def format_scientific_tag(value: float) -> str:
    formatted = f"{value:.0e}"
    mantissa, exponent = formatted.split("e")
    sign = exponent[0]
    power = int(exponent[1:])
    if sign == "-":
        return f"{mantissa}e-{power}"
    return f"{mantissa}e{power}"


def build_runtime_config(overrides: dict | None = None) -> RuntimeConfig:
    settings = {
        "NU": NU,
        "SOLVE_NX": SOLVE_NX,
        "SOLVE_NY": SOLVE_NY,
        "SAVE_NX": SAVE_NX,
        "SAVE_NY": SAVE_NY,
        "SOLVE_DT": SOLVE_DT,
        "SAVE_DT": SAVE_DT,
        "NUM_STEPS": NUM_STEPS,
        "NUM_SAMPLES": NUM_SAMPLES,
        "PARALLEL_CASES": PARALLEL_CASES,
        "PLOT": PLOT,
        "SEED": SEED,
        "SAVE_FIELDS": SAVE_FIELDS,
        "GRF_ALPHA": GRF_ALPHA,
        "GRF_TAU": GRF_TAU,
        "OUTPUT_ROOT": OUTPUT_ROOT,
    }
    if overrides:
        settings.update(overrides)
    output_root = Path(settings["OUTPUT_ROOT"])
    if not output_root.is_absolute():
        output_root = (Path(__file__).resolve().parent / output_root).resolve()
    config = RuntimeConfig(
        nu=float(settings["NU"]),
        solve_nx=int(settings["SOLVE_NX"]),
        solve_ny=int(settings["SOLVE_NY"]),
        save_nx=int(settings["SAVE_NX"]),
        save_ny=int(settings["SAVE_NY"]),
        solve_dt=float(settings["SOLVE_DT"]),
        save_dt=float(settings["SAVE_DT"]),
        num_steps=int(settings["NUM_STEPS"]),
        num_samples=int(settings["NUM_SAMPLES"]),
        parallel_cases=int(settings["PARALLEL_CASES"]),
        plot=bool(settings["PLOT"]),
        seed=int(settings["SEED"]),
        save_fields=normalize_save_fields(settings["SAVE_FIELDS"]),
        grf_alpha=float(settings["GRF_ALPHA"]),
        grf_tau=float(settings["GRF_TAU"]),
        output_root=output_root,
    )
    validate_runtime_config(config)
    return config


def validate_runtime_config(config: RuntimeConfig) -> None:
    if config.solve_nx != config.solve_ny:
        raise ValueError(
            "To keep the solver identical to the original FNO code, SOLVE_NX and SOLVE_NY must match."
        )
    if config.save_nx != config.save_ny:
        raise ValueError("For this script, SAVE_NX and SAVE_NY must match.")
    if config.solve_nx <= 0 or config.save_nx <= 0:
        raise ValueError("Spatial resolutions must be positive integers.")
    if config.solve_dt <= 0 or config.save_dt <= 0:
        raise ValueError("Time steps must be positive.")
    if config.num_steps < 2:
        raise ValueError(
            "NUM_STEPS must be at least 2 so that residual statistics are well defined."
        )
    if config.num_samples <= 0:
        raise ValueError("NUM_SAMPLES must be positive.")
    if config.parallel_cases <= 0:
        raise ValueError("PARALLEL_CASES must be positive.")
    if config.parallel_cases > config.num_samples:
        raise ValueError("PARALLEL_CASES cannot be larger than NUM_SAMPLES.")
    if config.save_fields not in {"omega", "uv", "uvomega"}:
        raise ValueError("SAVE_FIELDS must be one of {'omega', 'uv', 'uvomega'}.")
    ratio = config.save_dt / config.solve_dt
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("SAVE_DT / SOLVE_DT must be a positive integer.")
    if config.num_steps % config.save_stride != 0:
        raise ValueError("NUM_STEPS must be divisible by SAVE_DT / SOLVE_DT.")
    if config.solve_nx % config.save_nx != 0 or config.solve_ny % config.save_ny != 0:
        raise ValueError(
            "When saving at a lower spatial resolution, SOLVE_NX and SOLVE_NY must be divisible by SAVE_NX and SAVE_NY."
        )


def normalize_save_fields(value: str) -> str:
    normalized = str(value).strip().lower()
    alias_map = {
        "w": "omega",
        "omega": "omega",
        "uv": "uv",
        "u,v": "uv",
        "omega_uv": "uvomega",
        "uvomega": "uvomega",
        "omegauv": "uvomega",
        "uv_omega": "uvomega",
        "omega+uv": "uvomega",
        "uv+omega": "uvomega",
    }
    if normalized not in alias_map:
        raise ValueError("SAVE_FIELDS must be one of {'omega', 'uv', 'uvomega'}.")
    return alias_map[normalized]


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 16,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
        }
    )


class GaussianRF:

    def __init__(
        self,
        dim: int,
        size: int,
        alpha: float = 2.0,
        tau: float = 3.0,
        sigma: float | None = None,
        device: torch.device | None = None,
    ):
        if dim != 2:
            raise ValueError(
                "This single-file script currently implements only the 2D periodic Gaussian random field."
            )
        self.dim = dim
        self.size = size
        self.device = device if device is not None else torch.device("cpu")
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        wavenumbers = torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=self.device),
                torch.arange(start=-k_max, end=0, step=1, device=self.device),
            ),
            dim=0,
        ).repeat(size, 1)
        k_x = wavenumbers.transpose(0, 1).to(torch.float32)
        k_y = wavenumbers.to(torch.float32)
        sqrt_eig = size**2 * math.sqrt(2.0) * sigma
        sqrt_eig = sqrt_eig * (4 * math.pi**2 * (k_x**2 + k_y**2) + tau**2) ** (
            -alpha / 2.0
        )
        sqrt_eig[0, 0] = 0.0
        self.sqrt_eig = sqrt_eig

    def sample(self, num_samples: int) -> torch.Tensor:
        real_part = torch.randn(num_samples, self.size, self.size, device=self.device)
        imag_part = torch.randn(num_samples, self.size, self.size, device=self.device)
        coeff = torch.complex(real_part, imag_part)
        coeff = self.sqrt_eig.unsqueeze(0) * coeff
        return torch.fft.ifft2(coeff).real.to(torch.float32)


def build_forcing(grid_size: int, device: torch.device) -> torch.Tensor:
    grid = torch.linspace(0, 1, grid_size + 1, device=device, dtype=torch.float32)[:-1]
    x_coord, y_coord = torch.meshgrid(grid, grid, indexing="ij")
    return 0.1 * (
        torch.sin(2 * math.pi * (x_coord + y_coord))
        + torch.cos(2 * math.pi * (x_coord + y_coord))
    )


def build_spectral_operators(
    grid_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k_max = math.floor(grid_size / 2.0)
    k_y = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1, device=device),
            torch.arange(start=-k_max, end=0, step=1, device=device),
        ),
        dim=0,
    ).repeat(grid_size, 1)
    k_x = k_y.transpose(0, 1)
    k_x = k_x[..., : k_max + 1].to(torch.float32)
    k_y = k_y[..., : k_max + 1].to(torch.float32)
    lap = 4 * math.pi**2 * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0
    dealias = torch.logical_and(
        torch.abs(k_y) <= 2.0 / 3.0 * k_max, torch.abs(k_x) <= 2.0 / 3.0 * k_max
    ).float()
    return (k_x, k_y, lap, dealias.unsqueeze(0))


def downsample_spatial(
    tensor: torch.Tensor, save_nx: int, save_ny: int
) -> torch.Tensor:
    solve_nx = tensor.shape[-2]
    solve_ny = tensor.shape[-1]
    step_x = solve_nx // save_nx
    step_y = solve_ny // save_ny
    return tensor[..., ::step_x, ::step_y][..., :save_nx, :save_ny]


def central_difference_x(values: torch.Tensor, dx: float) -> torch.Tensor:
    return (
        torch.roll(values, shifts=-1, dims=-2) - torch.roll(values, shifts=1, dims=-2)
    ) / (2.0 * dx)


def central_difference_y(values: torch.Tensor, dy: float) -> torch.Tensor:
    return (
        torch.roll(values, shifts=-1, dims=-1) - torch.roll(values, shifts=1, dims=-1)
    ) / (2.0 * dy)


def central_laplacian(values: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    return (
        torch.roll(values, shifts=-1, dims=-2)
        - 2.0 * values
        + torch.roll(values, shifts=1, dims=-2)
    ) / (dx * dx) + (
        torch.roll(values, shifts=-1, dims=-1)
        - 2.0 * values
        + torch.roll(values, shifts=1, dims=-1)
    ) / (
        dy * dy
    )


def finite_difference_stream_function(vorticity: torch.Tensor) -> torch.Tensor:
    nx = vorticity.shape[-2]
    ny = vorticity.shape[-1]
    dx = 1.0 / nx
    dy = 1.0 / ny
    kx = torch.arange(nx, device=vorticity.device, dtype=vorticity.dtype)
    ky = torch.arange(ny, device=vorticity.device, dtype=vorticity.dtype)
    neg_lap_x = 4.0 * torch.sin(math.pi * kx / nx) ** 2 / (dx * dx)
    neg_lap_y = 4.0 * torch.sin(math.pi * ky / ny) ** 2 / (dy * dy)
    neg_lap = neg_lap_x[:, None] + neg_lap_y[None, :]
    neg_lap[0, 0] = 1.0
    psi_h = torch.fft.fft2(vorticity) / neg_lap
    psi_h[..., 0, 0] = 0.0
    return torch.fft.ifft2(psi_h).real


def finite_difference_navier_stokes_terms(
    vorticity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    nx = vorticity.shape[-2]
    ny = vorticity.shape[-1]
    dx = 1.0 / nx
    dy = 1.0 / ny
    psi = finite_difference_stream_function(vorticity)
    velocity_x = central_difference_y(psi, dy)
    velocity_y = -central_difference_x(psi, dx)
    vorticity_x = central_difference_x(vorticity, dx)
    vorticity_y = central_difference_y(vorticity, dy)
    nonlinear_term = velocity_x * vorticity_x + velocity_y * vorticity_y
    laplacian_w = central_laplacian(vorticity, dx, dy)
    return (nonlinear_term, laplacian_w)


def compute_velocity_from_vorticity(
    vorticity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    nx = vorticity.shape[-2]
    ny = vorticity.shape[-1]
    dx = 1.0 / nx
    dy = 1.0 / ny
    psi = finite_difference_stream_function(vorticity)
    velocity_x = central_difference_y(psi, dy)
    velocity_y = -central_difference_x(psi, dx)
    return (velocity_x, velocity_y)


def build_saved_export_from_vorticity(
    vorticity: torch.Tensor, config: RuntimeConfig
) -> torch.Tensor:
    save_fields = normalize_save_fields(config.save_fields)
    omega_saved = downsample_spatial(vorticity, config.save_nx, config.save_ny)
    if save_fields == "omega":
        return omega_saved
    velocity_x, velocity_y = compute_velocity_from_vorticity(vorticity)
    u_saved = downsample_spatial(velocity_x, config.save_nx, config.save_ny)
    v_saved = downsample_spatial(velocity_y, config.save_nx, config.save_ny)
    if save_fields == "uv":
        return torch.stack((u_saved, v_saved), dim=-1)
    return torch.stack((u_saved, v_saved, omega_saved), dim=-1)


def compute_nonlinear_and_laplacian(
    w_h: torch.Tensor,
    k_x: torch.Tensor,
    k_y: torch.Tensor,
    lap: torch.Tensor,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    psi_h = w_h / lap
    q = 2.0 * math.pi * k_y * 1j * psi_h
    q = torch.fft.irfft2(q, s=(grid_size, grid_size))
    v = -2.0 * math.pi * k_x * 1j * psi_h
    v = torch.fft.irfft2(v, s=(grid_size, grid_size))
    w_x = 2.0 * math.pi * k_x * 1j * w_h
    w_x = torch.fft.irfft2(w_x, s=(grid_size, grid_size))
    w_y = 2.0 * math.pi * k_y * 1j * w_h
    w_y = torch.fft.irfft2(w_y, s=(grid_size, grid_size))
    nonlinear_term = q * w_x + v * w_y
    laplacian_w = torch.fft.irfft2(-lap * w_h, s=(grid_size, grid_size))
    return (nonlinear_term, laplacian_w)


def solve_batch_navier_stokes(
    initial_vorticity_batch: torch.Tensor,
    forcing: torch.Tensor,
    config: RuntimeConfig,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    batch_size = initial_vorticity_batch.shape[0]
    grid_size = config.solve_size
    save_count = config.saved_times
    k_x, k_y, lap, dealias = build_spectral_operators(grid_size, device)
    w_h = torch.fft.rfft2(initial_vorticity_batch)
    f_h = torch.fft.rfft2(forcing)
    if f_h.ndim < w_h.ndim:
        f_h = f_h.unsqueeze(0)
    save_fields = normalize_save_fields(config.save_fields)
    saved_vorticity_trajectories = torch.empty(
        batch_size,
        config.save_nx,
        config.save_ny,
        save_count,
        dtype=torch.float32,
        device="cpu",
    )
    saved_vorticity_trajectories[..., 0] = (
        downsample_spatial(initial_vorticity_batch, config.save_nx, config.save_ny)
        .detach()
        .cpu()
    )
    if save_fields == "omega":
        saved_export_trajectories = saved_vorticity_trajectories
    else:
        initial_export = (
            build_saved_export_from_vorticity(initial_vorticity_batch, config)
            .detach()
            .cpu()
        )
        saved_export_trajectories = torch.empty(
            batch_size,
            config.save_nx,
            config.save_ny,
            save_count,
            int(initial_export.shape[-1]),
            dtype=torch.float32,
            device="cpu",
        )
        saved_export_trajectories[..., 0, :] = initial_export
    current_w = initial_vorticity_batch
    original_sum_abs = torch.zeros(batch_size, dtype=torch.float64, device=device)
    original_max_abs = torch.zeros(batch_size, dtype=torch.float32, device=device)
    original_count = 0
    iterator = range(config.num_steps)
    if show_progress:
        iterator = tqdm(iterator, total=config.num_steps)
    save_index = 1
    for step in iterator:
        nonlinear_term, laplacian_w = compute_nonlinear_and_laplacian(
            w_h, k_x, k_y, lap, grid_size
        )
        nonlinear_h = torch.fft.rfft2(nonlinear_term)
        nonlinear_h = dealias * nonlinear_h
        next_w_h = (
            -config.solve_dt * nonlinear_h
            + config.solve_dt * f_h
            + (1.0 - 0.5 * config.solve_dt * config.nu * lap) * w_h
        ) / (1.0 + 0.5 * config.solve_dt * config.nu * lap)
        next_w = torch.fft.irfft2(next_w_h, s=(grid_size, grid_size))
        fd_nonlinear_term, fd_laplacian_w = finite_difference_navier_stokes_terms(
            current_w
        )
        residual = (
            (next_w - current_w) / config.solve_dt
            + fd_nonlinear_term
            - config.nu * fd_laplacian_w
            - forcing.unsqueeze(0)
        )
        abs_residual = residual.abs()
        original_sum_abs += abs_residual.sum(dim=(-2, -1), dtype=torch.float64)
        original_max_abs = torch.maximum(
            original_max_abs, abs_residual.amax(dim=(-2, -1))
        )
        original_count += abs_residual.shape[-2] * abs_residual.shape[-1]
        if (step + 1) % config.save_stride == 0:
            saved_vorticity_trajectories[..., save_index] = (
                downsample_spatial(next_w, config.save_nx, config.save_ny)
                .detach()
                .cpu()
            )
            if save_fields != "omega":
                saved_export_trajectories[..., save_index, :] = (
                    build_saved_export_from_vorticity(next_w, config).detach().cpu()
                )
            save_index += 1
        current_w = next_w
        w_h = next_w_h
    mean_abs = (original_sum_abs / original_count).detach().cpu().numpy()
    max_abs = original_max_abs.detach().cpu().numpy()
    return (saved_vorticity_trajectories, saved_export_trajectories, mean_abs, max_abs)


def compute_residual_field_at_time(
    trajectory: torch.Tensor,
    forcing: torch.Tensor,
    viscosity: float,
    delta_t: float,
    time_index: int,
) -> torch.Tensor:
    total_times = trajectory.shape[-1]
    if time_index >= total_times - 1:
        raise ValueError(
            "Forward-difference residual is undefined at the final saved time."
        )
    current_w = trajectory[..., time_index]
    time_derivative = (trajectory[..., time_index + 1] - current_w) / delta_t
    nonlinear_term, laplacian_w = finite_difference_navier_stokes_terms(current_w)
    residual = time_derivative + nonlinear_term - viscosity * laplacian_w - forcing
    return residual


def compute_saved_resolution_residuals(
    saved_trajectory: np.ndarray,
    forcing_saved: np.ndarray,
    viscosity: float,
    save_dt: float,
    device: torch.device,
    plot_indices: list[int],
) -> tuple[float, float, list[np.ndarray]]:
    trajectory = torch.from_numpy(saved_trajectory).to(
        device=device, dtype=torch.float32
    )
    forcing = torch.from_numpy(forcing_saved).to(device=device, dtype=torch.float32)
    total_times = trajectory.shape[-1]
    if total_times < 2:
        raise ValueError(
            "At least two saved times are required to compute forward-difference residual statistics."
        )
    sum_abs = 0.0
    max_abs = 0.0
    count = 0
    chunk_size = 16
    for start_index in range(0, total_times - 1, chunk_size):
        end_index = min(total_times - 1, start_index + chunk_size)
        center = trajectory[..., start_index:end_index].permute(2, 0, 1).contiguous()
        next_values = (
            trajectory[..., start_index + 1 : end_index + 1]
            .permute(2, 0, 1)
            .contiguous()
        )
        time_derivative = (next_values - center) / save_dt
        nonlinear_term, laplacian_w = finite_difference_navier_stokes_terms(center)
        residual = (
            time_derivative
            + nonlinear_term
            - viscosity * laplacian_w
            - forcing.unsqueeze(0)
        )
        abs_residual = residual.abs()
        sum_abs += abs_residual.sum().item()
        max_abs = max(max_abs, abs_residual.max().item())
        count += abs_residual.numel()
    residual_fields = []
    for time_index in plot_indices:
        residual = compute_residual_field_at_time(
            trajectory, forcing, viscosity, save_dt, time_index
        )
        residual_fields.append(residual.abs().detach().cpu().numpy().astype(np.float32))
    return (sum_abs / count, max_abs, residual_fields)


def choose_plot_indices(total_times: int) -> list[int]:
    raw_indices = [
        0,
        round((total_times - 1) / 3),
        round(2 * (total_times - 1) / 3),
        total_times - 1,
    ]
    ordered = []
    for index in raw_indices:
        if index not in ordered:
            ordered.append(index)
    while len(ordered) < 4:
        ordered.append(total_times - 1)
    return ordered[:4]


def choose_residual_plot_indices(
    vorticity_plot_indices: list[int], total_times: int
) -> list[int]:
    if total_times < 2:
        raise ValueError(
            "At least two saved times are required to plot forward-difference residuals."
        )
    last_valid_residual_index = total_times - 2
    return [min(index, last_valid_residual_index) for index in vorticity_plot_indices]


def save_trajectory_file(trajectory: np.ndarray, output_path: Path) -> float:
    np.save(output_path, trajectory.astype(np.float32))
    return output_path.stat().st_size / (1024.0 * 1024.0)


def save_info_file(
    output_path: Path,
    sample_wall_time_sec: float,
    nu: float,
    save_fields: str,
    original_mean_abs_residual: float,
    original_max_abs_residual: float,
    saved_mean_abs_residual: float,
    saved_max_abs_residual: float,
    trajectory_file_size_mib: float,
) -> None:
    re_value = 1.0 / nu
    lines = [
        f"sample_wall_time_sec: {sample_wall_time_sec:.6f}",
        f"nu: {nu:.6e}",
        f"save_fields: {save_fields}",
        f"Re_equal_1_over_nu: {re_value:.6e}",
        f"mean_abs_residual_solve_resolution: {original_mean_abs_residual:.6e}",
        f"max_abs_residual_solve_resolution: {original_max_abs_residual:.6e}",
        f"mean_abs_residual_save_resolution: {saved_mean_abs_residual:.6e}",
        f"max_abs_residual_save_resolution: {saved_max_abs_residual:.6e}",
        f"trajectory_file_size_mib: {trajectory_file_size_mib:.6f}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(
    output_path: Path,
    trajectory: np.ndarray,
    residual_fields: list[np.ndarray],
    vorticity_plot_indices: list[int],
    residual_plot_indices: list[int],
    save_dt: float,
) -> None:
    configure_matplotlib_style()
    vorticity_slices = [trajectory[..., index] for index in vorticity_plot_indices]
    vorticity_min = min((slice_value.min() for slice_value in vorticity_slices))
    vorticity_max = max((slice_value.max() for slice_value in vorticity_slices))
    residual_max = max((field.max() for field in residual_fields))
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for column, (time_index, slice_value) in enumerate(
        zip(vorticity_plot_indices, vorticity_slices)
    ):
        axis = axes[0, column]
        image = axis.imshow(
            slice_value,
            cmap="RdBu_r",
            origin="lower",
            vmin=vorticity_min,
            vmax=vorticity_max,
        )
        axis.set_title(f"$\\omega(x, y, t={time_index * save_dt:.4f})$")
        axis.set_xticks([])
        axis.set_yticks([])
    vorticity_colorbar = figure.colorbar(image, ax=axes[0, :], shrink=0.85)
    vorticity_colorbar.ax.set_ylabel("$\\omega$")
    for column, (time_index, residual_field) in enumerate(
        zip(residual_plot_indices, residual_fields)
    ):
        axis = axes[1, column]
        image = axis.imshow(
            residual_field, cmap="magma", origin="lower", vmin=0.0, vmax=residual_max
        )
        axis.set_title(f"$|R|(x, y, t={time_index * save_dt:.4f})$")
        axis.set_xticks([])
        axis.set_yticks([])
    residual_colorbar = figure.colorbar(image, ax=axes[1, :], shrink=0.85)
    residual_colorbar.ax.set_ylabel("$|R|$")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def generate_dataset(config: RuntimeConfig, *, overwrite: bool = False) -> Path:
    config.output_root.mkdir(parents=True, exist_ok=True)
    parameter_directory = config.parameter_directory
    if parameter_directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {parameter_directory}; pass --overwrite to regenerate."
            )
        shutil.rmtree(parameter_directory)
    parameter_directory.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    set_random_seed(config.seed)
    forcing_solve = build_forcing(config.solve_size, device)
    forcing_save = (
        downsample_spatial(forcing_solve, config.save_nx, config.save_ny)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    grf = GaussianRF(
        2, config.solve_size, alpha=config.grf_alpha, tau=config.grf_tau, device=device
    )
    channels = {"omega": ["omega"], "uv": ["u", "v"], "uvomega": ["u", "v", "omega"]}[
        config.save_fields
    ]
    generation_metadata = {
        "nu": config.nu,
        "solve_nx": config.solve_nx,
        "solve_ny": config.solve_ny,
        "save_nx": config.save_nx,
        "save_ny": config.save_ny,
        "solve_dt": config.solve_dt,
        "save_dt": config.save_dt,
        "num_steps": config.num_steps,
        "num_samples": config.num_samples,
        "saved_times": config.saved_times,
        "seed": config.seed,
        "parallel_cases": config.parallel_cases,
        "grf_alpha": config.grf_alpha,
        "grf_tau": config.grf_tau,
        "save_fields": config.save_fields,
        "channel_order": channels,
        "torch_version": torch.__version__,
        "device": str(device),
        "domain": "[0,1)^2",
        "boundary": "periodic",
        "forcing": "0.1*(sin(2*pi*(x+y))+cos(2*pi*(x+y)))",
        "velocity_export": "finite-difference stream-function solve and centered differences on solver grid, then index subsampling",
        "omega_export": "solver vorticity, then index subsampling",
    }
    (parameter_directory / "generation_config.json").write_text(
        json.dumps(generation_metadata, indent=2) + "\n", encoding="utf-8"
    )
    sample_index = 0
    while sample_index < config.num_samples:
        batch_size = min(config.parallel_cases, config.num_samples - sample_index)
        initial_batch = grf.sample(batch_size).to(torch.float32)
        synchronize(device)
        batch_start = time.perf_counter()
        (
            saved_batch_vorticity,
            saved_batch_export,
            original_mean_abs,
            original_max_abs,
        ) = solve_batch_navier_stokes(
            initial_batch, forcing_solve, config, device, show_progress=batch_size == 1
        )
        synchronize(device)
        batch_wall_time_sec = time.perf_counter() - batch_start
        for local_index in range(batch_size):
            sample_number = sample_index + local_index + 1
            sample_directory = parameter_directory / f"{sample_number:04d}"
            if sample_directory.exists():
                shutil.rmtree(sample_directory)
            sample_directory.mkdir(parents=True, exist_ok=True)
            sample_start = time.perf_counter()
            sample_vorticity_trajectory = (
                saved_batch_vorticity[local_index].numpy().astype(np.float32)
            )
            sample_export_trajectory = (
                saved_batch_export[local_index].numpy().astype(np.float32)
            )
            vorticity_plot_indices = choose_plot_indices(
                sample_vorticity_trajectory.shape[-1]
            )
            residual_plot_indices = choose_residual_plot_indices(
                vorticity_plot_indices, sample_vorticity_trajectory.shape[-1]
            )
            saved_mean_abs, saved_max_abs, residual_fields = (
                compute_saved_resolution_residuals(
                    saved_trajectory=sample_vorticity_trajectory,
                    forcing_saved=forcing_save,
                    viscosity=config.nu,
                    save_dt=config.save_dt,
                    device=device,
                    plot_indices=residual_plot_indices,
                )
            )
            trajectory_path = sample_directory / "trajectory.npy"
            file_size_mib = save_trajectory_file(
                sample_export_trajectory, trajectory_path
            )
            if config.plot:
                save_plot(
                    output_path=sample_directory / "plot.png",
                    trajectory=sample_vorticity_trajectory,
                    residual_fields=residual_fields,
                    vorticity_plot_indices=vorticity_plot_indices,
                    residual_plot_indices=residual_plot_indices,
                    save_dt=config.save_dt,
                )
            sample_postprocess_sec = time.perf_counter() - sample_start
            sample_wall_time_sec = (
                batch_wall_time_sec / batch_size + sample_postprocess_sec
            )
            save_info_file(
                output_path=sample_directory / "info.txt",
                sample_wall_time_sec=sample_wall_time_sec,
                nu=config.nu,
                save_fields=config.save_fields,
                original_mean_abs_residual=float(original_mean_abs[local_index]),
                original_max_abs_residual=float(original_max_abs[local_index]),
                saved_mean_abs_residual=float(saved_mean_abs),
                saved_max_abs_residual=float(saved_max_abs),
                trajectory_file_size_mib=file_size_mib,
            )
        sample_index += batch_size
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return parameter_directory.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate periodic NS trajectories with explicit channel order."
    )
    overrides = {
        "nu": ("NU", float),
        "solve_nx": ("SOLVE_NX", int),
        "solve_ny": ("SOLVE_NY", int),
        "save_nx": ("SAVE_NX", int),
        "save_ny": ("SAVE_NY", int),
        "solve_dt": ("SOLVE_DT", float),
        "save_dt": ("SAVE_DT", float),
        "num_steps": ("NUM_STEPS", int),
        "num_samples": ("NUM_SAMPLES", int),
        "parallel_cases": ("PARALLEL_CASES", int),
        "seed": ("SEED", int),
        "grf_alpha": ("GRF_ALPHA", float),
        "grf_tau": ("GRF_TAU", float),
        "save_fields": ("SAVE_FIELDS", str),
        "output_root": ("OUTPUT_ROOT", str),
    }
    for name, (_, value_type) in overrides.items():
        parser.add_argument("--" + name, type=value_type, default=None)
    parser.add_argument("--plot", action="store_true", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = {
        key: getattr(args, name)
        for name, (key, _) in overrides.items()
        if getattr(args, name) is not None
    }
    if args.plot is not None:
        settings["PLOT"] = args.plot
    config = build_runtime_config(settings)
    output_relative_path = generate_dataset(config, overwrite=args.overwrite)
    print(f"Generated dataset directory: {output_relative_path}")


if __name__ == "__main__":
    main()
