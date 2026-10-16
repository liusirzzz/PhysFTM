"""
3D heat equation solver (finite difference, explicit Euler).

    u_t = alpha * (u_xx + u_yy + u_zz)

Domain   : [0, L]^3, uniform grid N x N x N (default 64^3)
BC       : homogeneous Dirichlet on all 6 faces (non-periodic)
Scheme   : 2nd-order central difference in space, forward Euler in time
Stability: dt <= h^2 / (6*alpha), h = L/(N-1)

Initial condition has NO known exact solution (Gaussian hot blob);
the script only saves the snapshots and plots.

Outputs (saved in ./output):
    heat3d_solution.npz  - x, times, snapshots (n_snapshots, N, N, N)
    fig_slices.png       - mid-plane slices at several times
    fig_decay.png        - max value and total "heat" vs time
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")


def initial_condition(X, Y, Z):
    """Default IC: Gaussian hot blob at the center. No exact solution needed."""
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    return np.exp(-r2 / 0.02)


def solve_heat3d(
    alpha=0.01,
    N=64,
    L=1.0,
    T=4.0,
    n_snapshots=100,
    ic_func=initial_condition,
    safety=0.9,
    verbose=True,
):
    """
    Solve u_t = alpha * Laplacian(u) on [0,L]^3 with u=0 on the boundary.

    Parameters
    ----------
    alpha       : diffusion coefficient
    N           : grid points per direction (grid is N x N x N)
    L           : domain side length
    T           : final time
    n_snapshots : number of saved snapshots, uniformly spaced in time
                  (including t=0 and t=T)
    ic_func     : callable f(X, Y, Z) -> initial field (should satisfy BC)
    safety      : CFL safety factor (< 1)

    Returns
    -------
    x           : (N,) grid coordinates
    times       : (n_snapshots,) snapshot times
    snapshots   : (n_snapshots, N, N, N) saved solutions (float32)
    """
    h = L / (N - 1)
    dt_max = h**2 / (6.0 * alpha)
    dt = safety * dt_max
    nsteps = int(np.ceil(T / dt))
    dt = T / nsteps
    r = alpha * dt / h**2
    if verbose:
        print(
            f"h = {h:.6e}, dt = {dt:.6e} (limit {dt_max:.6e}), r = {r:.4f} (<= 1/6 = {1 / 6:.4f}), steps = {nsteps}"
        )
    x = np.linspace(0.0, L, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = ic_func(X, Y, Z).astype(np.float64)
    u[0, :, :] = u[-1, :, :] = 0.0
    u[:, 0, :] = u[:, -1, :] = 0.0
    u[:, :, 0] = u[:, :, -1] = 0.0
    times = np.linspace(0.0, T, n_snapshots)
    save_steps = np.round(times / dt).astype(int)
    times = save_steps * dt
    snapshots = np.empty((n_snapshots, N, N, N), dtype=np.float32)
    snap_i = 0
    if save_steps[0] == 0:
        snapshots[0] = u
        snap_i = 1
    for n in range(1, nsteps + 1):
        lap = (
            u[2:, 1:-1, 1:-1]
            + u[:-2, 1:-1, 1:-1]
            + u[1:-1, 2:, 1:-1]
            + u[1:-1, :-2, 1:-1]
            + u[1:-1, 1:-1, 2:]
            + u[1:-1, 1:-1, :-2]
            - 6.0 * u[1:-1, 1:-1, 1:-1]
        ) / h**2
        u[1:-1, 1:-1, 1:-1] += alpha * dt * lap
        while snap_i < n_snapshots and save_steps[snap_i] == n:
            snapshots[snap_i] = u
            snap_i += 1
    return (x, times, snapshots)
