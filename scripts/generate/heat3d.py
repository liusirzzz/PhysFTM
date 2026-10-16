"""
Dataset generator for the 3D heat equation solver.

Each sample = one random initial condition (sum of random Gaussian blobs)
solved with solve_heat3d(). Solutions are saved one .npz per sample under
./dataset/, and the IC parameters of every sample are logged to
./dataset/metadata.csv for reproducibility.

Multiprocessing is used to generate samples in parallel.

Config is at the bottom of the file (N_SAMPLES, N_SNAPSHOTS, SAVE_DTYPE, ...).
"""

import csv
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from heat3d_solver import solve_heat3d

DATASET_DIR = Path("data/raw/heat3d")


def make_random_ic(seed, n_blobs=3, margin=0.25):
    """
    Build an initial condition f(X, Y, Z) from `n_blobs` Gaussian blobs
    with random centers (kept `margin` away from the boundary so that the
    field is ~0 on the Dirichlet boundary), random widths and amplitudes.

    Returns (ic_func, params) where params is a flat dict for the metadata.
    """
    rng = np.random.default_rng(seed)
    centers = rng.uniform(margin, 1.0 - margin, size=(n_blobs, 3))
    widths = rng.uniform(0.005, 0.03, size=n_blobs)
    amps = rng.uniform(0.5, 1.5, size=n_blobs)

    def ic(X, Y, Z):
        u = np.zeros_like(X)
        for (cx, cy, cz), w, a in zip(centers, widths, amps):
            u += a * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / w)
        return u

    params = {"seed": seed, "n_blobs": n_blobs}
    for i in range(n_blobs):
        params[f"blob{i}_cx"] = centers[i, 0]
        params[f"blob{i}_cy"] = centers[i, 1]
        params[f"blob{i}_cz"] = centers[i, 2]
        params[f"blob{i}_width"] = widths[i]
        params[f"blob{i}_amp"] = amps[i]
    return (ic, params)


def generate_one(args):
    """args = (seed, cfg). Returns (seed, metadata dict)."""
    seed, cfg = args
    ic, params = make_random_ic(seed, n_blobs=cfg["n_blobs"])
    x, times, U = solve_heat3d(
        alpha=cfg["alpha"],
        N=cfg["N"],
        L=cfg["L"],
        T=cfg["T"],
        n_snapshots=cfg["n_snapshots"],
        ic_func=ic,
        verbose=False,
    )
    path = DATASET_DIR / f"sample_{seed:04d}.npz"
    np.savez_compressed(
        path,
        x=x.astype(np.float32),
        times=times,
        snapshots=U.astype(cfg["save_dtype"]),
        alpha=cfg["alpha"],
        N=cfg["N"],
        L=cfg["L"],
        T=cfg["T"],
    )
    params["file"] = path.name
    params["size_mb"] = round(path.stat().st_size / 1000000.0, 2)
    return (seed, params)


if __name__ == "__main__":
    cfg = {
        "alpha": 0.01,
        "N": 64,
        "L": 1.0,
        "T": 4.0,
        "n_snapshots": 50,
        "n_blobs": 3,
        "save_dtype": np.float32,
    }
    N_SAMPLES = 200
    N_WORKERS = 16
    SEED0 = 0
    DATASET_DIR.mkdir(exist_ok=True)
    jobs = [(SEED0 + i, cfg) for i in range(N_SAMPLES)]
    with Pool(N_WORKERS) as pool:
        results = pool.map(generate_one, jobs)
    results.sort(key=lambda r: r[0])
    meta_path = DATASET_DIR / "metadata.csv"
    fields = list(results[0][1].keys())
    with open(meta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for _, params in results:
            w.writerow(params)
    total_mb = sum((p["size_mb"] for _, p in results))
    print(f"\nGenerated {len(results)} samples in {DATASET_DIR}")
    for seed, p in results:
        print(f"  sample_{seed:04d}.npz  {p['size_mb']:6.2f} MB")
    print(f"Total: {total_mb:.1f} MB -> about {total_mb / len(results):.1f} MB/sample")
    print(f"Metadata: {meta_path}")
