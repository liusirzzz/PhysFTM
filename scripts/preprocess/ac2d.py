import argparse
import json
import re
from pathlib import Path
import h5py
import numpy as np

DEFAULT_SOURCE_DIR = Path("data/raw/ac2d")
DEFAULT_OUTPUT_DIR = Path("data/processed/ac2d")
FILE_PATTERN = re.compile("^IBVP2_s(\\d+)_n(\\d+)\\.npz$")
SUPPORTED_GRID_SIZES = (32, 64, 128, 256)


def parse_sample_file(path: Path) -> tuple[int, int]:
    match = FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected filename format: {path}")
    return (int(match.group(1)), int(match.group(2)))


def collect_npz_files(root: Path, grid_size: int) -> list[Path]:
    files = []
    for path in root.glob(f"IBVP2_s*_n{grid_size}.npz"):
        sample_idx, parsed_grid = parse_sample_file(path)
        if parsed_grid != grid_size:
            continue
        files.append((sample_idx, path))
    files = [path for _, path in sorted(files, key=lambda item: item[0])]
    if not files:
        raise FileNotFoundError(
            f"No IBVP2_s*_n{grid_size}.npz files found under {root}"
        )
    sample_indices = [parse_sample_file(path)[0] for path in files]
    expected = list(range(len(files)))
    if sample_indices != expected:
        raise ValueError(
            f"Sample indices for grid_size={grid_size} are not contiguous from 0. First few indices: {sample_indices[:10]}"
        )
    return files


def load_reference_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path) as npz:
        for key in ("tspan", "xx", "yy"):
            if key not in npz:
                raise KeyError(f"{npz_path} missing '{key}'")
        tspan = np.asarray(npz["tspan"], dtype=np.float32)
        xx = np.asarray(npz["xx"], dtype=np.float32)
        yy = np.asarray(npz["yy"], dtype=np.float32)
    return (tspan, xx, yy)


def load_solution(npz_path: Path, dtype: np.dtype) -> np.ndarray:
    with np.load(npz_path) as npz:
        if "solution" not in npz:
            raise KeyError(f"{npz_path} missing 'solution'")
        solution = np.asarray(npz["solution"], dtype=dtype)
    if solution.ndim != 3:
        raise ValueError(f"{npz_path} solution must be (H,W,T), got {solution.shape}")
    return np.transpose(solution, (2, 0, 1))


def make_per_sequence_time_consistent_mask(
    rng: np.random.RandomState,
    num_samples: int,
    time_steps: int,
    height: int,
    width: int,
    ratio: float,
) -> np.ndarray:
    spatial_mask = (rng.rand(num_samples, 1, height, width, 1) < ratio).astype(np.uint8)
    return np.repeat(spatial_mask, time_steps, axis=1)


def format_ratio_tag(ratio: float) -> str:
    return format(float(ratio), "g")


def build_eval_seed(base_seed: int, eval_mask_ratio: float) -> int:
    ratio_tag = format_ratio_tag(eval_mask_ratio).replace(".", "").replace("-", "m")
    return int(base_seed) * 1000 + int(ratio_tag)


def ensure_clean_output(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and (not overwrite):
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            + "\n".join((str(path) for path in existing))
        )


def process_ac_data(
    source_dir: Path,
    out_dir: Path,
    grid_size: int,
    split_ratios: tuple[float, float, float],
    seed: int,
    train_mask_ratio: float,
    eval_mask_ratios: tuple[float, ...],
    dtype: np.dtype,
    max_samples: int | None,
    overwrite: bool,
    metadata_only: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = collect_npz_files(source_dir, grid_size)
    full_count = len(files)
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise ValueError("No source samples found after applying max_samples.")
    sample0 = load_solution(files[0], dtype=dtype)
    tspan, xx, yy = load_reference_arrays(files[0])
    time_steps, height, width = sample0.shape
    if height != grid_size or width != grid_size:
        raise ValueError(
            f"grid_size={grid_size} does not match loaded sample shape {(height, width)} from {files[0]}"
        )
    if xx.shape != (height, width) or yy.shape != (height, width):
        raise ValueError(
            f"Coordinate arrays must match spatial shape {(height, width)}, got xx={xx.shape}, yy={yy.shape}"
        )
    if tspan.shape != (time_steps,):
        raise ValueError(f"tspan shape must be ({time_steps},), got {tspan.shape}")
    num_samples = len(files)
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"split_ratios must sum to 1, got {split_ratios}")
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError(
            f"Split sizes must all be positive, got train={num_train}, val={num_val}, test={num_test}. Increase max_samples or change split_ratios."
        )
    eval_mask_ratios = tuple((float(ratio) for ratio in eval_mask_ratios))
    if not eval_mask_ratios:
        raise ValueError("eval_mask_ratios must contain at least one ratio.")
    for ratio in eval_mask_ratios:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"eval mask ratio must be in (0, 1], got {ratio}")
    prefix = f"AC_n{grid_size}_temporal_consistent_mask"
    split_suffix = (
        f"tr{num_train}_val{num_val}_te{num_test}_trmr{train_mask_ratio}_seed{seed}"
    )
    train_data_path = out_dir / f"{prefix}_tr_data_{split_suffix}.h5"
    val_data_path = out_dir / f"{prefix}_val_data_{split_suffix}.h5"
    test_data_path = out_dir / f"{prefix}_te_data_{split_suffix}.npy"
    output_paths = []
    if not metadata_only:
        output_paths.extend([train_data_path, val_data_path, test_data_path])
    for eval_mask_ratio in eval_mask_ratios:
        ratio_tag = format_ratio_tag(eval_mask_ratio)
        eval_suffix = f"{split_suffix}_evmr{ratio_tag}"
        output_paths.extend(
            [
                out_dir / f"{prefix}_tr_metadata_{eval_suffix}.npy",
                out_dir / f"{prefix}_val_metadata_{eval_suffix}.npy",
                out_dir / f"{prefix}_te_metadata_{eval_suffix}.npy",
                out_dir / f"{prefix}_manifest_{eval_suffix}.json",
            ]
        )
    ensure_clean_output(output_paths, overwrite=overwrite)
    if metadata_only:
        missing_data_paths = [
            path
            for path in (train_data_path, val_data_path, test_data_path)
            if not path.exists()
        ]
        if missing_data_paths:
            raise FileNotFoundError(
                "metadata_only=True requires existing split data files:\n"
                + "\n".join((str(path) for path in missing_data_paths))
            )
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_samples)
    indices_tr = indices[:num_train]
    indices_val = indices[num_train : num_train + num_val]
    indices_te = indices[num_train + num_val :]
    train_pos = {int(src_idx): pos for pos, src_idx in enumerate(indices_tr)}
    val_pos = {int(src_idx): pos for pos, src_idx in enumerate(indices_val)}
    test_pos = {int(src_idx): pos for pos, src_idx in enumerate(indices_te)}
    if not metadata_only:
        test_data = np.empty((num_test, time_steps, height, width), dtype=dtype)
        chunk_shape = (1, time_steps, height, width)
        with h5py.File(train_data_path, "w") as train_h5, h5py.File(
            val_data_path, "w"
        ) as val_h5:
            train_ds = train_h5.create_dataset(
                "data",
                shape=(num_train, time_steps, height, width),
                dtype=dtype,
                chunks=chunk_shape,
            )
            val_ds = val_h5.create_dataset(
                "data",
                shape=(num_val, time_steps, height, width),
                dtype=dtype,
                chunks=chunk_shape,
            )
            for src_idx, npz_path in enumerate(files):
                sample = (
                    sample0 if src_idx == 0 else load_solution(npz_path, dtype=dtype)
                )
                if sample.shape != sample0.shape:
                    raise ValueError(
                        f"Inconsistent sample shape for {npz_path}: expected {sample0.shape}, got {sample.shape}"
                    )
                if src_idx in train_pos:
                    train_ds[train_pos[src_idx]] = sample
                elif src_idx in val_pos:
                    val_ds[val_pos[src_idx]] = sample
                elif src_idx in test_pos:
                    test_data[test_pos[src_idx]] = sample
                else:
                    raise RuntimeError(
                        f"Sample index {src_idx} was not assigned to any split."
                    )
                if (src_idx + 1) % 25 == 0 or src_idx + 1 == num_samples:
                    print(f"[n{grid_size}] processed {src_idx + 1}/{num_samples}")
        np.save(test_data_path, {"data": test_data})
    else:
        print(
            f"[n{grid_size}] metadata_only=True | reuse existing data files:\n  train={train_data_path}\n  val={val_data_path}\n  test={test_data_path}"
        )
    u_ind_uni = np.linspace(0, height - 1, height).astype(np.float32) / (height - 1)
    v_ind_uni = np.linspace(0, width - 1, width).astype(np.float32) / (width - 1)
    t_ind_uni = np.linspace(0, time_steps - 1, time_steps).astype(np.float32) / (
        time_steps - 1
    )
    x_coords_physical = xx[0].astype(np.float32)
    y_coords_physical = yy[:, 0].astype(np.float32)
    t_coords_physical = tspan.astype(np.float32)
    train_mask_rng = np.random.RandomState(seed)
    mask_tr = make_per_sequence_time_consistent_mask(
        rng=train_mask_rng,
        num_samples=num_train,
        time_steps=time_steps,
        height=height,
        width=width,
        ratio=train_mask_ratio,
    )
    common_meta = {
        "u_ind_uni": u_ind_uni,
        "v_ind_uni": v_ind_uni,
        "t_ind_uni": t_ind_uni,
        "u_coords_physical": x_coords_physical,
        "v_coords_physical": y_coords_physical,
        "t_coords_physical": t_coords_physical,
        "split_ratios": np.asarray(split_ratios, dtype=np.float32),
        "seed": seed,
        "indices_tr": indices_tr,
        "indices_val": indices_val,
        "indices_te": indices_te,
        "train_mask_ratio": train_mask_ratio,
        "consistent_points_across_time": True,
        "source_dir": str(source_dir),
        "grid_size": int(grid_size),
        "full_sample_count": int(full_count),
        "used_sample_count": int(num_samples),
    }
    manifest_index = {}
    for eval_mask_ratio in eval_mask_ratios:
        ratio_tag = format_ratio_tag(eval_mask_ratio)
        eval_seed = build_eval_seed(seed, eval_mask_ratio)
        eval_rng = np.random.RandomState(eval_seed)
        mask_val = make_per_sequence_time_consistent_mask(
            rng=eval_rng,
            num_samples=num_val,
            time_steps=time_steps,
            height=height,
            width=width,
            ratio=eval_mask_ratio,
        )
        mask_te = make_per_sequence_time_consistent_mask(
            rng=eval_rng,
            num_samples=num_test,
            time_steps=time_steps,
            height=height,
            width=width,
            ratio=eval_mask_ratio,
        )
        eval_suffix = f"{split_suffix}_evmr{ratio_tag}"
        train_meta_path = out_dir / f"{prefix}_tr_metadata_{eval_suffix}.npy"
        val_meta_path = out_dir / f"{prefix}_val_metadata_{eval_suffix}.npy"
        test_meta_path = out_dir / f"{prefix}_te_metadata_{eval_suffix}.npy"
        manifest_path = out_dir / f"{prefix}_manifest_{eval_suffix}.json"
        meta_common = {
            **common_meta,
            "eval_mask_ratio": float(eval_mask_ratio),
            "eval_mask_seed": int(eval_seed),
        }
        np.save(train_meta_path, {"data": {**meta_common, "mask_tr": mask_tr}})
        np.save(val_meta_path, {"data": {**meta_common, "mask_val": mask_val}})
        np.save(test_meta_path, {"data": {**meta_common, "mask_te": mask_te}})
        manifest = {
            "data_name": prefix,
            "dim": "2d",
            "split_ratios": list(map(float, split_ratios)),
            "seed": seed,
            "grid_size": int(grid_size),
            "num_samples": int(num_samples),
            "time_steps": int(time_steps),
            "train_mask_ratio": float(train_mask_ratio),
            "eval_mask_ratio": float(eval_mask_ratio),
            "consistent_points_across_time": True,
            "source_dir": str(source_dir),
            "splits": {
                "train": {
                    "data_path": str(train_data_path),
                    "metadata_path": str(train_meta_path),
                },
                "val": {
                    "data_path": str(val_data_path),
                    "metadata_path": str(val_meta_path),
                },
                "test": {
                    "data_path": str(test_data_path),
                    "metadata_path": str(test_meta_path),
                },
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        manifest_index[ratio_tag] = str(manifest_path)
    print(
        f"[n{grid_size}] done | train={num_train} val={num_val} test={num_test} | eval_mask_ratios={list(eval_mask_ratios)}"
    )
    for ratio_tag, manifest_path in manifest_index.items():
        print(f"[n{grid_size}] evmr={ratio_tag} manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare AC2D data in the RBC/SW manifest format."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs="+",
        default=[256],
        choices=SUPPORTED_GRID_SIZES,
        help="One or more source spatial resolutions to export.",
    )
    parser.add_argument("--split_ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_mask_ratio", type=float, default=0.15)
    parser.add_argument("--eval_mask_ratio", type=float, default=0.03)
    parser.add_argument(
        "--eval_mask_ratios",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of val/test mask ratios. If set, overrides --eval_mask_ratio.",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Reuse existing split data files under output_dir and only generate metadata/manifest.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dtype = np.dtype(args.dtype)
    eval_mask_ratios = (
        tuple((float(x) for x in args.eval_mask_ratios))
        if args.eval_mask_ratios is not None
        else (float(args.eval_mask_ratio),)
    )
    for grid_size in args.grid_size:
        process_ac_data(
            source_dir=args.source_dir,
            out_dir=args.output_dir,
            grid_size=int(grid_size),
            split_ratios=tuple((float(x) for x in args.split_ratios)),
            seed=int(args.seed),
            train_mask_ratio=float(args.train_mask_ratio),
            eval_mask_ratios=eval_mask_ratios,
            dtype=dtype,
            max_samples=args.max_samples,
            overwrite=bool(args.overwrite),
            metadata_only=bool(args.metadata_only),
        )


if __name__ == "__main__":
    main()
