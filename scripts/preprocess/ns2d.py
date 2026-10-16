import argparse
import json
import re
import sys
from pathlib import Path
import h5py
import numpy as np

DEFAULT_SOURCE_DIR = Path("data/raw/ns2d/nu_1e-4_shape_128x128x201_dt_1e-1")
DEFAULT_OUTPUT_DIR = Path("data/processed/ns2d")
DEFAULT_EVAL_MASK_RATIOS = (0.15, 0.1, 0.05, 0.03, 0.01)
SAMPLE_DIR_PATTERN = re.compile("^\\d+$")
SOURCE_DIR_PATTERN = re.compile(
    "^nu_(?P<nu>[^_]+)_shape_(?P<h>\\d+)x(?P<w>\\d+)x(?P<t>\\d+)_dt_(?P<dt>[^_]+?)(?:_fields_(?P<fields>.+))?$"
)


def collect_sample_dirs(root: Path) -> list[Path]:
    sample_dirs = []
    for path in root.iterdir():
        if not path.is_dir() or SAMPLE_DIR_PATTERN.match(path.name) is None:
            continue
        trajectory_path = path / "trajectory.npy"
        if trajectory_path.exists():
            sample_dirs.append(path)
    sample_dirs = sorted(sample_dirs, key=lambda path: int(path.name))
    if not sample_dirs:
        raise FileNotFoundError(
            f"No sample directories with trajectory.npy found under {root}"
        )
    return sample_dirs


def parse_source_dir_info(source_dir: Path) -> dict[str, float | int | None]:
    match = SOURCE_DIR_PATTERN.match(source_dir.name)
    if match is None:
        return {
            "nu": None,
            "dt": None,
            "height": None,
            "width": None,
            "time_steps": None,
        }
    return {
        "nu": float(match.group("nu")),
        "dt": float(match.group("dt")),
        "height": int(match.group("h")),
        "width": int(match.group("w")),
        "time_steps": int(match.group("t")),
    }


def load_trajectory(
    sample_dir: Path, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    trajectory_path = sample_dir / "trajectory.npy"
    trajectory = np.asarray(np.load(trajectory_path), dtype=dtype)
    if trajectory.ndim != 4 or trajectory.shape[-1] != 3:
        raise ValueError(
            f"NS2D requires (H,W,T,3) in (u,v,omega) order, got {trajectory.shape}"
        )
    uvomega = np.transpose(trajectory, (2, 0, 1, 3))
    return (uvomega, uvomega[..., 2], ("u", "v", "omega"), "uvomega")


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


def process_ns2d_data(
    source_dir: Path,
    out_dir: Path,
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
    sample_dirs = collect_sample_dirs(source_dir)
    full_count = len(sample_dirs)
    if max_samples is not None:
        sample_dirs = sample_dirs[:max_samples]
    if not sample_dirs:
        raise ValueError("No source samples found after applying max_samples.")
    sample0, sample0_omega, channel_order, resolved_field_mode = load_trajectory(
        sample_dirs[0], dtype=dtype
    )
    source_info = parse_source_dir_info(source_dir)
    if sample0.ndim == 3:
        time_steps, height, width = sample0.shape
        num_channels = 1
    elif sample0.ndim == 4:
        time_steps, height, width, num_channels = sample0.shape
    else:
        raise ValueError(
            f"Processed trajectory must be (T,H,W) or (T,H,W,C), got {sample0.shape}"
        )
    if source_info["height"] is not None and (
        height != source_info["height"] or width != source_info["width"]
    ):
        raise ValueError(
            f"Source dir shape tag {(source_info['height'], source_info['width'])} does not match sample shape {(height, width)}"
        )
    if (
        source_info["time_steps"] is not None
        and time_steps != source_info["time_steps"]
    ):
        raise ValueError(
            f"Source dir time_steps tag {source_info['time_steps']} does not match sample shape {time_steps}"
        )
    num_samples = len(sample_dirs)
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
    nu_tag = "unknown" if source_info["nu"] is None else format(source_info["nu"], "g")
    dt_tag = "unknown" if source_info["dt"] is None else format(source_info["dt"], "g")
    field_tag = "" if resolved_field_mode == "omega" else f"_{resolved_field_mode}"
    prefix = f"NS2D{field_tag}_n{height}_t{time_steps}_nu{nu_tag}_dt{dt_tag}_temporal_consistent_mask"
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
        data_shape = (
            (time_steps, height, width)
            if num_channels == 1
            else (time_steps, height, width, num_channels)
        )
        test_data = np.empty((num_test, *data_shape), dtype=dtype)
        chunk_shape = (1, *data_shape)
        with h5py.File(train_data_path, "w") as train_h5, h5py.File(
            val_data_path, "w"
        ) as val_h5:
            train_ds = train_h5.create_dataset(
                "data", shape=(num_train, *data_shape), dtype=dtype, chunks=chunk_shape
            )
            val_ds = val_h5.create_dataset(
                "data", shape=(num_val, *data_shape), dtype=dtype, chunks=chunk_shape
            )
            for src_idx, sample_dir in enumerate(sample_dirs):
                if src_idx == 0:
                    sample = sample0
                else:
                    sample, _, _, _ = load_trajectory(sample_dir, dtype=dtype)
                if sample.shape != sample0.shape:
                    raise ValueError(
                        f"Inconsistent sample shape for {sample_dir}: expected {sample0.shape}, got {sample.shape}"
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
                if (src_idx + 1) % 50 == 0 or src_idx + 1 == num_samples:
                    print(f"processed {src_idx + 1}/{num_samples}")
        np.save(test_data_path, {"data": test_data})
    else:
        print(
            f"metadata_only=True | reuse existing data files:\n  train={train_data_path}\n  val={val_data_path}\n  test={test_data_path}"
        )
    u_ind_uni = np.linspace(0, height - 1, height).astype(np.float32) / (height - 1)
    v_ind_uni = np.linspace(0, width - 1, width).astype(np.float32) / (width - 1)
    t_ind_uni = np.linspace(0, time_steps - 1, time_steps).astype(np.float32) / (
        time_steps - 1
    )
    x_coords_physical = np.arange(height, dtype=np.float32) / float(height)
    y_coords_physical = np.arange(width, dtype=np.float32) / float(width)
    dt = 1.0 if source_info["dt"] is None else float(source_info["dt"])
    t_coords_physical = (np.arange(time_steps, dtype=np.float32) * dt).astype(
        np.float32
    )
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
        "grid_size": int(height),
        "time_steps": int(time_steps),
        "full_sample_count": int(full_count),
        "used_sample_count": int(num_samples),
        "nu": source_info["nu"],
        "dt": dt,
        "field_name": "trajectory",
        "field_mode": resolved_field_mode,
        "num_channels": int(num_channels),
        "channel_order": np.asarray(channel_order, dtype=object),
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
            "grid_size": int(height),
            "num_samples": int(num_samples),
            "time_steps": int(time_steps),
            "field_mode": resolved_field_mode,
            "num_channels": int(num_channels),
            "channel_order": list(channel_order),
            "train_mask_ratio": float(train_mask_ratio),
            "eval_mask_ratio": float(eval_mask_ratio),
            "consistent_points_across_time": True,
            "source_dir": str(source_dir),
            "nu": source_info["nu"],
            "dt": dt,
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
        f"done | train={num_train} val={num_val} test={num_test} | eval_mask_ratios={list(eval_mask_ratios)}"
    )
    for ratio_tag, manifest_path in manifest_index.items():
        print(f"evmr={ratio_tag} manifest={manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process NS2D trajectory data into project raw_data format."
    )
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split_ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_mask_ratio", type=float, default=0.15)
    parser.add_argument("--eval_mask_ratio", type=float, default=0.03)
    parser.add_argument(
        "--eval_mask_ratios",
        type=float,
        nargs="+",
        default=None,
        help="List of val/test mask ratios. Defaults to 0.15 0.1 0.05 0.03 0.01.",
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
    if args.eval_mask_ratios is not None:
        eval_mask_ratios = tuple((float(x) for x in args.eval_mask_ratios))
    elif "--eval_mask_ratio" in sys.argv:
        eval_mask_ratios = (float(args.eval_mask_ratio),)
    else:
        eval_mask_ratios = DEFAULT_EVAL_MASK_RATIOS
    process_ns2d_data(
        source_dir=args.source_dir,
        out_dir=args.output_dir,
        split_ratios=tuple((float(x) for x in args.split_ratios)),
        seed=int(args.seed),
        train_mask_ratio=float(args.train_mask_ratio),
        eval_mask_ratios=eval_mask_ratios,
        dtype=np.dtype(args.dtype),
        max_samples=args.max_samples,
        overwrite=bool(args.overwrite),
        metadata_only=bool(args.metadata_only),
    )


if __name__ == "__main__":
    main()
