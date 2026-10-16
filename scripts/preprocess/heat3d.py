"""Prepare the generated Heat3D samples in the project's raw-data format.

The source directory contains one compressed ``sample_XXXX.npz`` per random
initial condition.  This script writes one shared set of train/validation/test
data files and metadata/manifest files for both temporal-consistent and
time-varying observation masks.

Example::

    python scripts/heat3d_dataProcess.py

The default output is ``raw_data/Heat3D``.  Data are stored as
``(sample, time, x, y, z)``; masks use the same shape without a channel axis.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
import h5py
import numpy as np

DEFAULT_SOURCE_DIR = Path("data/raw/heat3d")
DEFAULT_OUTPUT_DIR = Path("data/processed/heat3d")
DEFAULT_EVAL_MASK_RATIOS = (0.01, 0.03, 0.05, 0.1, 0.15)
SAMPLE_PATTERN = re.compile("^sample_(\\d+)\\.npz$")
MASK_MODES = ("temporal_consistent",)


def ratio_tag(ratio: float) -> str:
    return format(float(ratio), "g")


def eval_seed(base_seed: int, ratio: float) -> int:
    tag = ratio_tag(ratio).replace(".", "").replace("-", "m")
    return int(base_seed) * 1000 + int(tag)


def collect_samples(source_dir: Path) -> list[Path]:
    pairs: list[tuple[int, Path]] = []
    for path in source_dir.glob("sample_*.npz"):
        match = SAMPLE_PATTERN.match(path.name)
        if match is not None:
            pairs.append((int(match.group(1)), path))
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        raise FileNotFoundError(f"No sample_*.npz files found under {source_dir}")
    indices = [index for index, _ in pairs]
    expected = list(range(len(pairs)))
    if indices != expected:
        raise ValueError(
            f"Sample indices must be contiguous from zero; got first indices {indices[:10]}"
        )
    return [path for _, path in pairs]


def scalar_from_npz(npz: Any, key: str, path: Path) -> float | int:
    if key not in npz:
        raise KeyError(f"{path} missing '{key}'")
    value = np.asarray(npz[key])
    if value.ndim != 0:
        raise ValueError(f"{path} '{key}' must be scalar, got {value.shape}")
    item = value.item()
    return int(item) if key == "N" else float(item)


def load_sample(
    path: Path, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    with np.load(path) as npz:
        for key in ("x", "times", "snapshots", "alpha", "N", "L", "T"):
            if key not in npz:
                raise KeyError(f"{path} missing '{key}'")
        x = np.asarray(npz["x"], dtype=np.float32)
        times = np.asarray(npz["times"], dtype=np.float32)
        snapshots = np.asarray(npz["snapshots"], dtype=dtype)
        parameters = {
            "alpha": scalar_from_npz(npz, "alpha", path),
            "N": scalar_from_npz(npz, "N", path),
            "L": scalar_from_npz(npz, "L", path),
            "T": scalar_from_npz(npz, "T", path),
        }
    if x.ndim != 1:
        raise ValueError(f"{path} x must be (N,), got {x.shape}")
    if times.ndim != 1:
        raise ValueError(f"{path} times must be (T,), got {times.shape}")
    if snapshots.ndim != 4:
        raise ValueError(f"{path} snapshots must be (T,N,N,N), got {snapshots.shape}")
    n = int(parameters["N"])
    if snapshots.shape[1:] != (n, n, n) or x.shape != (n,):
        raise ValueError(
            f"{path} has inconsistent N={n}, x={x.shape}, snapshots={snapshots.shape}"
        )
    if snapshots.shape[0] != times.size:
        raise ValueError(
            f"{path} times/snapshots mismatch: {times.shape} vs {snapshots.shape}"
        )
    return (snapshots, x, times, parameters)


def assert_close_arrays(
    reference: np.ndarray, value: np.ndarray, name: str, path: Path
) -> None:
    if reference.shape != value.shape or not np.allclose(
        reference, value, rtol=1e-06, atol=1e-06
    ):
        raise ValueError(
            f"{path} has inconsistent {name}: expected {reference.shape}, got {value.shape}"
        )


def inspect_source(
    files: list[Path], dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    sample0, x, times, parameters = load_sample(files[0], dtype=dtype)
    for path in files[1:]:
        sample, current_x, current_times, current_parameters = load_sample(
            path, dtype=dtype
        )
        if sample.shape != sample0.shape:
            raise ValueError(
                f"{path} shape {sample.shape} differs from {sample0.shape}"
            )
        assert_close_arrays(x, current_x, "x coordinates", path)
        assert_close_arrays(times, current_times, "time coordinates", path)
        for key, expected in parameters.items():
            if key == "N":
                same = int(current_parameters[key]) == int(expected)
            else:
                same = bool(
                    np.isclose(
                        float(current_parameters[key]),
                        float(expected),
                        rtol=1e-06,
                        atol=1e-08,
                    )
                )
            if not same:
                raise ValueError(
                    f"{path} parameter {key} differs: expected {expected}, got {current_parameters[key]}"
                )
    return (sample0, x, times, parameters)


def split_indices(
    num_samples: int, split_ratios: tuple[float, float, float], seed: int
) -> tuple[np.ndarray, ...]:
    if not np.isclose(sum(split_ratios), 1.0):
        raise ValueError(f"split_ratios must sum to 1, got {split_ratios}")
    if any((ratio < 0 for ratio in split_ratios)):
        raise ValueError(f"split_ratios must be non-negative, got {split_ratios}")
    permutation = np.random.RandomState(seed).permutation(num_samples)
    num_train = int(num_samples * split_ratios[0])
    num_val = int(num_samples * split_ratios[1])
    num_test = num_samples - num_train - num_val
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError(
            f"Split sizes must be positive, got {num_train}/{num_val}/{num_test}"
        )
    return (
        permutation[:num_train],
        permutation[num_train : num_train + num_val],
        permutation[num_train + num_val :],
    )


def ensure_clean(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and (not overwrite):
        raise FileExistsError(
            "Output files already exist; use --overwrite:\n"
            + "\n".join(map(str, existing))
        )


def write_split_data(
    files: list[Path],
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    data_paths: tuple[Path, Path, Path],
    sample_shape: tuple[int, ...],
    dtype: np.dtype,
    overwrite: bool,
) -> None:
    train_indices, val_indices, test_indices = indices
    positions = {
        int(source): ("train", position)
        for position, source in enumerate(train_indices)
    }
    positions.update(
        {int(source): ("val", position) for position, source in enumerate(val_indices)}
    )
    positions.update(
        {
            int(source): ("test", position)
            for position, source in enumerate(test_indices)
        }
    )
    train_path, val_path, test_path = data_paths
    temporary_paths = [path.with_name(path.name + ".tmp") for path in data_paths]
    for path in temporary_paths:
        path.unlink(missing_ok=True)
    try:
        with h5py.File(temporary_paths[0], "w") as train_h5, h5py.File(
            temporary_paths[1], "w"
        ) as val_h5:
            train_ds = train_h5.create_dataset(
                "data",
                shape=(len(train_indices), *sample_shape),
                dtype=dtype,
                chunks=(1, *sample_shape),
            )
            val_ds = val_h5.create_dataset(
                "data",
                shape=(len(val_indices), *sample_shape),
                dtype=dtype,
                chunks=(1, *sample_shape),
            )
            test_data = np.empty((len(test_indices), *sample_shape), dtype=dtype)
            for source_index, path in enumerate(files):
                sample, _, _, _ = load_sample(path, dtype=dtype)
                split, position = positions[source_index]
                if split == "train":
                    train_ds[position] = sample
                elif split == "val":
                    val_ds[position] = sample
                else:
                    test_data[position] = sample
                if (source_index + 1) % 25 == 0 or source_index + 1 == len(files):
                    print(f"processed {source_index + 1}/{len(files)} samples")
        with temporary_paths[2].open("wb") as handle:
            np.save(handle, {"data": test_data}, allow_pickle=True)
        for temporary, target in zip(temporary_paths, data_paths):
            os.replace(temporary, target)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def make_mask(
    rng: np.random.RandomState,
    num_samples: int,
    time_steps: int,
    spatial_shape: tuple[int, int, int],
    ratio: float,
    mode: str,
) -> np.ndarray:
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"Mask ratio must be in (0, 1], got {ratio}")
    total = int(np.prod(spatial_shape))
    count = int(round(ratio * total))
    mask = np.zeros((num_samples, time_steps, *spatial_shape), dtype=np.uint8)
    for sample_index in range(num_samples):
        times = (0,)
        for time_index in times:
            selected = rng.choice(total, size=count, replace=False)
            mask[sample_index, time_index].reshape(-1)[selected] = 1
        if mode == "temporal_consistent" and time_steps > 1:
            mask[sample_index, 1:] = mask[sample_index, :1]
    return mask


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, metadata, allow_pickle=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def provenance_copy(source_dir: Path, output_dir: Path, overwrite: bool) -> Path:
    source = source_dir / "metadata.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing provenance file: {source}")
    target = output_dir / "heat3d_source_metadata.csv"
    if target.exists() and (not overwrite):
        return target
    shutil.copyfile(source, target)
    return target


def manifest_paths(
    output_dir: Path,
    mode: str,
    data_paths: tuple[Path, Path, Path],
    split_counts: tuple[int, int, int],
    train_ratio: float,
    eval_ratio: float,
    seed: int,
    grid_size: int,
    time_steps: int,
) -> tuple[Path, Path, Path, Path]:
    prefix = f"Heat3D_n{grid_size}_t{time_steps}_{mode}_mask"
    tr, val, te = split_counts
    suffix = f"tr{tr}_val{val}_te{te}_trmr{ratio_tag(train_ratio)}_seed{seed}_evmr{ratio_tag(eval_ratio)}"
    return (
        output_dir / f"{prefix}_tr_metadata_{suffix}.npy",
        output_dir / f"{prefix}_val_metadata_{suffix}.npy",
        output_dir / f"{prefix}_te_metadata_{suffix}.npy",
        output_dir / f"{prefix}_manifest_{suffix}.json",
    )


def process_heat3d(
    source_dir: Path,
    output_dir: Path,
    split_ratios: tuple[float, float, float],
    seed: int,
    train_mask_ratio: float,
    eval_mask_ratios: tuple[float, ...],
    mask_modes: tuple[str, ...],
    dtype: np.dtype,
    max_samples: int | None,
    overwrite: bool,
    metadata_only: bool,
    dry_run: bool,
) -> None:
    files = collect_samples(source_dir)
    full_sample_count = len(files)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        files = files[:max_samples]
    if not files:
        raise ValueError("No samples remain after applying max_samples")
    sample0, x, times, parameters = inspect_source(files, dtype=dtype)
    num_samples = len(files)
    train_indices, val_indices, test_indices = split_indices(
        num_samples, split_ratios, seed
    )
    split_indices_tuple = (train_indices, val_indices, test_indices)
    split_counts = (len(train_indices), len(val_indices), len(test_indices))
    grid_size = int(sample0.shape[1])
    time_steps = int(sample0.shape[0])
    if sample0.shape[1:] != (grid_size, grid_size, grid_size):
        raise ValueError(f"Heat3D samples must be cubic, got {sample0.shape}")
    if int(parameters["N"]) != grid_size:
        raise ValueError(
            f"N={parameters['N']} disagrees with snapshots shape {sample0.shape}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = (
        provenance_copy(source_dir, output_dir, overwrite=overwrite)
        if not dry_run
        else output_dir / "heat3d_source_metadata.csv"
    )
    data_prefix = f"Heat3D_n{grid_size}_t{time_steps}"
    data_paths = (
        output_dir
        / f"{data_prefix}_tr_data_tr{split_counts[0]}_val{split_counts[1]}_te{split_counts[2]}_seed{seed}.h5",
        output_dir
        / f"{data_prefix}_val_data_tr{split_counts[0]}_val{split_counts[1]}_te{split_counts[2]}_seed{seed}.h5",
        output_dir
        / f"{data_prefix}_te_data_tr{split_counts[0]}_val{split_counts[1]}_te{split_counts[2]}_seed{seed}.npy",
    )
    if not metadata_only and (not dry_run):
        ensure_clean(list(data_paths), overwrite=overwrite)
        write_split_data(
            files,
            split_indices_tuple,
            data_paths,
            tuple(sample0.shape),
            dtype=dtype,
            overwrite=overwrite,
        )
    elif metadata_only:
        missing = [path for path in data_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "metadata_only requires existing data files:\n"
                + "\n".join(map(str, missing))
            )
    u_ind_uni = np.linspace(0, 1, grid_size, dtype=np.float32)
    t_ind_uni = np.linspace(0, 1, time_steps, dtype=np.float32)
    common_meta: dict[str, Any] = {
        "u_ind_uni": u_ind_uni,
        "v_ind_uni": u_ind_uni.copy(),
        "w_ind_uni": u_ind_uni.copy(),
        "t_ind_uni": t_ind_uni,
        "u_coords_physical": x,
        "v_coords_physical": x.copy(),
        "w_coords_physical": x.copy(),
        "t_coords_physical": times,
        "split_ratios": np.asarray(split_ratios, dtype=np.float32),
        "seed": int(seed),
        "indices_tr": train_indices,
        "indices_val": val_indices,
        "indices_te": test_indices,
        "train_mask_ratio": float(train_mask_ratio),
        "source_dir": str(source_dir.resolve()),
        "provenance_csv": str(provenance_path.resolve()),
        "field_name": "temperature",
        "grid_size": grid_size,
        "time_steps": time_steps,
        "spatial_shape": np.asarray((grid_size, grid_size, grid_size), dtype=np.int64),
        "full_sample_count": int(full_sample_count),
        "used_sample_count": int(num_samples),
        "alpha": float(parameters["alpha"]),
        "L": float(parameters["L"]),
        "T": float(parameters["T"]),
        "dtype": str(dtype),
    }
    for mode in mask_modes:
        for eval_ratio in eval_mask_ratios:
            metadata_paths = manifest_paths(
                output_dir,
                mode,
                data_paths,
                split_counts,
                train_mask_ratio,
                eval_ratio,
                seed,
                grid_size,
                time_steps,
            )
            if dry_run:
                print("would write:")
                for path in (*metadata_paths,):
                    print(f"  {path}")
                continue
            ensure_clean(list(metadata_paths), overwrite=overwrite)
            train_mask = make_mask(
                np.random.RandomState(seed),
                len(train_indices),
                time_steps,
                (grid_size, grid_size, grid_size),
                train_mask_ratio,
                mode,
            )
            eval_rng = np.random.RandomState(eval_seed(seed, eval_ratio))
            val_mask = make_mask(
                eval_rng,
                len(val_indices),
                time_steps,
                (grid_size, grid_size, grid_size),
                eval_ratio,
                mode,
            )
            test_mask = make_mask(
                eval_rng,
                len(test_indices),
                time_steps,
                (grid_size, grid_size, grid_size),
                eval_ratio,
                mode,
            )
            mode_consistent = mode == "temporal_consistent"
            mode_meta = {
                **common_meta,
                "eval_mask_ratio": float(eval_ratio),
                "eval_mask_seed": int(eval_seed(seed, eval_ratio)),
                "mask_mode": mode,
                "mask_sampling": "fixed_count_uniform_random_spatial",
                "consistent_points_across_time": mode_consistent,
            }
            save_metadata(
                metadata_paths[0], {"data": {**mode_meta, "mask_tr": train_mask}}
            )
            save_metadata(
                metadata_paths[1], {"data": {**mode_meta, "mask_val": val_mask}}
            )
            save_metadata(
                metadata_paths[2], {"data": {**mode_meta, "mask_te": test_mask}}
            )
            manifest = {
                "data_name": f"Heat3D_n{grid_size}_t{time_steps}_{mode}_mask",
                "dim": "3d",
                "field_name": "temperature",
                "grid_size": grid_size,
                "time_steps": time_steps,
                "spatial_shape": [grid_size, grid_size, grid_size],
                "split_ratios": list(map(float, split_ratios)),
                "seed": int(seed),
                "num_samples": int(num_samples),
                "train_mask_ratio": float(train_mask_ratio),
                "eval_mask_ratio": float(eval_ratio),
                "mask_mode": mode,
                "mask_sampling": "fixed_count_uniform_random_spatial",
                "consistent_points_across_time": mode_consistent,
                "alpha": float(parameters["alpha"]),
                "L": float(parameters["L"]),
                "T": float(parameters["T"]),
                "source_dir": str(source_dir.resolve()),
                "provenance_csv": str(provenance_path.resolve()),
                "splits": {
                    "train": {
                        "data_path": str(data_paths[0].resolve()),
                        "metadata_path": str(metadata_paths[0].resolve()),
                    },
                    "val": {
                        "data_path": str(data_paths[1].resolve()),
                        "metadata_path": str(metadata_paths[1].resolve()),
                    },
                    "test": {
                        "data_path": str(data_paths[2].resolve()),
                        "metadata_path": str(metadata_paths[2].resolve()),
                    },
                },
            }
            with metadata_paths[3].open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
            print(f"wrote {metadata_paths[3]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split_ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_mask_ratio", type=float, default=0.15)
    parser.add_argument(
        "--eval_mask_ratios", type=float, nargs="+", default=DEFAULT_EVAL_MASK_RATIOS
    )
    parser.add_argument(
        "--mask_modes", choices=MASK_MODES, nargs="+", default=MASK_MODES
    )
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    process_heat3d(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split_ratios=tuple((float(value) for value in args.split_ratios)),
        seed=int(args.seed),
        train_mask_ratio=float(args.train_mask_ratio),
        eval_mask_ratios=tuple((float(value) for value in args.eval_mask_ratios)),
        mask_modes=tuple(args.mask_modes),
        dtype=np.dtype(args.dtype),
        max_samples=args.max_samples,
        overwrite=bool(args.overwrite),
        metadata_only=bool(args.metadata_only),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
