"""
Process ocean velocity field data from multiple years into 100 samples.
Each sample is 90 days, split 8:1:1 by temporal order.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List
import h5py
import netCDF4 as nc
import numpy as np

DEFAULT_LAT_RANGE = (-2.6666667461395264, 2.5833332538604736)
DEFAULT_LON_RANGE = (167.3333282470703, 172.5833282470703)
REFERENCE_DEPTH_VALUES = np.asarray(
    [
        0.49402499198913574,
        1.5413750410079956,
        2.6456689834594727,
        3.8194949626922607,
        5.078224182128906,
        6.440614223480225,
        7.92956018447876,
        9.572997093200684,
        11.404999732971191,
        13.467140197753906,
        15.810070037841797,
        18.495559692382812,
        21.598819732666016,
        25.211410522460938,
        29.444730758666992,
        34.43415069580078,
        40.344051361083984,
        47.37369155883789,
        55.76428985595703,
        65.80726623535156,
        77.85385131835938,
        92.3260726928711,
        109.72930145263672,
        130.66600036621094,
        155.85069274902344,
        186.12559509277344,
        222.47520446777344,
        266.0403137207031,
        318.1274108886719,
        380.2130126953125,
        453.9377136230469,
        541.0889282226562,
    ],
    dtype=np.float32,
)


def select_exact_region(coords, bounds, expected_size, axis_name):
    """Select the inclusive released crop; never resize or recenter it."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    low, high = map(float, bounds)
    if low >= high or expected_size <= 0:
        raise ValueError(f"Invalid {axis_name} bounds/size: {bounds}, {expected_size}")
    tolerance = 2e-05
    selected = np.flatnonzero(
        (coords >= low - tolerance) & (coords <= high + tolerance)
    )
    if selected.size != expected_size or np.any(np.diff(selected) != 1):
        raise ValueError(
            f"{axis_name}: bounds {bounds} select {selected.size} points; expected {expected_size} contiguous points. Check the source grid/config."
        )
    values = coords[selected]
    if not np.all(np.diff(values) > 0):
        raise ValueError(f"{axis_name} must increase, matching the released dataset.")
    if not np.allclose(values[[0, -1]], [low, high], rtol=0, atol=tolerance):
        raise ValueError(f"{axis_name} endpoints do not match the configured crop.")
    return (int(selected[0]), int(selected[-1]) + 1, values.astype(np.float32))


def select_depth_values(depth, num_depths):
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    if not 1 < num_depths <= depth.size:
        raise ValueError(
            f"Requested {num_depths} depth layers from {depth.size} available layers."
        )
    selected = depth[:num_depths]
    if num_depths == len(REFERENCE_DEPTH_VALUES) and (
        not np.allclose(selected, REFERENCE_DEPTH_VALUES, rtol=1e-06, atol=1e-06)
    ):
        raise ValueError(
            "Source first 32 depth coordinates differ from the released manifest."
        )
    return selected


def make_per_sequence_time_consistent_mask(
    rng: np.random.RandomState,
    num_samples: int,
    time_steps: int,
    height: int,
    width: int,
    ratio: float,
) -> np.ndarray:
    """Generate time-consistent spatial masks for each sample."""
    spatial_mask = (rng.rand(num_samples, 1, height, width, 1) < ratio).astype(np.uint8)
    return np.repeat(spatial_mask, time_steps, axis=1)


def format_ratio_tag(ratio: float) -> str:
    """Format ratio for filename."""
    return format(float(ratio), "g")


def build_eval_seed(base_seed: int, eval_mask_ratio: float) -> int:
    """Build evaluation seed from base seed and mask ratio."""
    ratio_tag = format_ratio_tag(eval_mask_ratio).replace(".", "").replace("-", "m")
    return int(base_seed) * 1000 + int(ratio_tag)


def ensure_clean_output(paths: List[Path], overwrite: bool) -> None:
    """Check if output files exist and handle overwrite logic."""
    existing = [path for path in paths if path.exists()]
    if existing and (not overwrite):
        raise FileExistsError(
            "Output files already exist. Use --overwrite to replace them:\n"
            + "\n".join((str(path) for path in existing))
        )


def find_nc_files(data_dir: Path) -> List[Path]:
    """Find all NetCDF files in chronological order."""
    nc_files = sorted(data_dir.glob("uv_fulldepth_*.nc"))
    print(f"Found {len(nc_files)} NetCDF files")
    for f in nc_files[:5]:
        print(f"  {f.name}")
    if len(nc_files) > 5:
        print(f"  ... and {len(nc_files) - 5} more")
    return nc_files


def read_daily_dates(ds, source_file):
    """Validate the actual CF time axis, including the chronological filename."""
    time_var = ds.variables["time"]
    decoded = nc.num2date(
        time_var[:],
        units=time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
    )
    if any(
        ((b - a).total_seconds() != 86400 for a, b in zip(decoded[:-1], decoded[1:]))
    ):
        raise ValueError(f"Time axis must be consecutive daily records: {source_file}")
    dates = [value.strftime("%Y-%m-%d") for value in decoded]
    match = re.fullmatch(
        "uv_fulldepth_(\\d{4}-\\d{2}-\\d{2})_to_(\\d{4}-\\d{2}-\\d{2})\\.nc",
        Path(source_file).name,
    )
    if match and dates and ((dates[0], dates[-1]) != match.groups()):
        raise ValueError(f"Filename dates differ from actual time axis: {source_file}")
    return dates


def plan_samples(nc_files, num_samples, days_per_sample, validate_grid=None):
    """Take full nonoverlapping blocks within each file, never across files."""
    plan, records = ([], [])
    for source_file in nc_files:
        if len(plan) >= num_samples:
            break
        with nc.Dataset(source_file) as ds:
            if validate_grid is not None:
                validate_grid(ds, source_file)
            dates = read_daily_dates(ds, source_file)
        for start in range(0, len(dates) - days_per_sample + 1, days_per_sample):
            if len(plan) >= num_samples:
                break
            stop, index = (start + days_per_sample, len(plan))
            plan.append((index, source_file, start, stop))
            records.append(
                dict(
                    sample_index=index,
                    source_file=str(source_file),
                    time_start_index=start,
                    time_stop_index_exclusive=stop,
                    start_date=dates[start],
                    end_date=dates[stop - 1],
                    dates=dates[start:stop],
                )
            )
    if len(plan) < num_samples:
        raise ValueError(f"Only found {len(plan)} complete samples, need {num_samples}")
    return (plan, records)


def extract_samples(
    nc_files: List[Path],
    num_samples: int,
    days_per_sample: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    target_size: tuple[int, int],
    num_workers: int = 8,
    num_depths: int = 32,
) -> tuple[np.ndarray, dict]:
    """
    Extract consecutive 90-day samples from multiple NetCDF files in parallel.
    Returns:
        samples: (N, T, D, H, W, 2) array where N=num_samples, T=days_per_sample
        metadata: dict with coordinate information
    """
    target_h, target_w = target_size
    total_days_needed = num_samples * days_per_sample
    print(
        f"\nNeed {total_days_needed} days total for {num_samples} samples of {days_per_sample} days each"
    )
    with nc.Dataset(nc_files[0]) as ds0:
        lat = ds0.variables["latitude"][:]
        lon = ds0.variables["longitude"][:]
        depth = ds0.variables["depth"][:]
        lat_start, lat_end, lat_selected = select_exact_region(
            lat, lat_range, target_h, "latitude"
        )
        lon_start, lon_end, lon_selected = select_exact_region(
            lon, lon_range, target_w, "longitude"
        )
        depth_selected = select_depth_values(depth, num_depths)
        dy = np.abs(np.mean(np.diff(lat_selected)))
        dx = np.abs(np.mean(np.diff(lon_selected)))
        metadata = {
            "lat_range": (float(lat_selected[0]), float(lat_selected[-1])),
            "lon_range": (float(lon_selected[0]), float(lon_selected[-1])),
            "depth_values": depth_selected.tolist(),
            "num_depths": len(depth_selected),
            "lat_values": lat_selected.tolist(),
            "lon_values": lon_selected.tolist(),
            "dx": float(dx),
            "dy": float(dy),
            "lat_start": lat_start,
            "lat_end": lat_end,
            "lon_start": lon_start,
            "lon_end": lon_end,
        }
    print(f"Region: lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]")
    print(
        f"Lat range: {metadata['lat_range'][0]:.2f}° to {metadata['lat_range'][1]:.2f}°"
    )
    print(
        f"Lon range: {metadata['lon_range'][0]:.2f}° to {metadata['lon_range'][1]:.2f}°"
    )

    def validate_grid(ds, nc_file):
        if not np.array_equal(
            np.asarray(ds.variables["latitude"][lat_start:lat_end], dtype=np.float32),
            lat_selected,
        ):
            raise ValueError(f"Latitude grid changed in {nc_file}")
        if not np.array_equal(
            np.asarray(ds.variables["longitude"][lon_start:lon_end], dtype=np.float32),
            lon_selected,
        ):
            raise ValueError(f"Longitude grid changed in {nc_file}")
        current_depth = select_depth_values(ds.variables["depth"][:], num_depths)
        if not np.array_equal(current_depth, depth_selected):
            raise ValueError(f"Depth grid changed in {nc_file}")

    sample_plan, sample_records = plan_samples(
        nc_files, num_samples, days_per_sample, validate_grid
    )
    metadata["sample_records"] = sample_records
    metadata["source_files_used"] = list(
        dict.fromkeys((str(p[1]) for p in sample_plan))
    )
    print(
        f"Planned {len(sample_plan)} samples across {len(set((p[1] for p in sample_plan)))} files"
    )
    region = (lat_start, lat_end, lon_start, lon_end, num_depths)
    read_args = [(idx, str(f), ts, te, region) for idx, f, ts, te in sample_plan]
    all_samples = [None] * num_samples
    completed = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"\nReading samples with {num_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_read_one_sample, arg): arg[0] for arg in read_args}
        for future in as_completed(futures):
            idx, sample = future.result()
            all_samples[idx] = sample
            completed += 1
            if completed % 10 == 0 or completed == num_samples:
                print(f"  {completed}/{num_samples} samples read")
    all_samples = np.stack(all_samples, axis=0)
    print(f"\nCollected {num_samples} samples with shape: {all_samples.shape}")
    return (all_samples, metadata)


def _read_one_sample(arg):
    """Worker function: read a single sample from a NetCDF file.
    netCDF4 auto-unpacks int16 -> float and returns a masked array where
    land/below-seafloor cells (source _FillValue = -32767) are masked.
    We fill those missing cells with 0.0 (no flow) so the output is a clean
    float field centered at 0, matching the reference NS2D data (no fill
    sentinels leaking into the tensor).
    """
    idx, nc_file, time_start, time_end, region = arg
    lat_start, lat_end, lon_start, lon_end, num_depths = region
    with nc.Dataset(nc_file) as ds:
        u_data = ds.variables["uo"][
            time_start:time_end, :num_depths, lat_start:lat_end, lon_start:lon_end
        ]
        v_data = ds.variables["vo"][
            time_start:time_end, :num_depths, lat_start:lat_end, lon_start:lon_end
        ]
    u_data = np.ma.filled(u_data, 0.0)
    v_data = np.ma.filled(v_data, 0.0)
    sample = np.stack([u_data, v_data], axis=-1).astype(np.float32)
    return (idx, sample)


def process_multi_year_data(
    data_dir: Path,
    out_dir: Path,
    num_samples: int,
    days_per_sample: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    target_size: tuple[int, int],
    split_ratios: tuple[float, float, float],
    seed: int,
    train_mask_ratio: float,
    eval_mask_ratios: tuple[float, ...],
    overwrite: bool,
    num_workers: int = 8,
    num_depths: int = 32,
) -> None:
    """Process multi-year ocean data into train/val/test splits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_files = find_nc_files(data_dir)
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {data_dir}")
    all_samples, metadata = extract_samples(
        nc_files,
        num_samples,
        days_per_sample,
        lat_range,
        lon_range,
        target_size,
        num_workers=num_workers,
        num_depths=num_depths,
    )
    num_samples_actual, time_steps, num_depths, height, width, num_channels = (
        all_samples.shape
    )
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"split_ratios must sum to 1, got {split_ratios}")
    num_train = int(num_samples_actual * train_ratio)
    num_val = int(num_samples_actual * val_ratio)
    num_test = num_samples_actual - num_train - num_val
    print(f"\nSplit: train={num_train}, val={num_val}, test={num_test}")
    train_data = all_samples[:num_train]
    val_data = all_samples[num_train : num_train + num_val]
    test_data = all_samples[num_train + num_val :]
    prefix = f"OCEAN_uv_n{height}_t{time_steps}_d{num_depths}_temporal_consistent_mask"
    split_suffix = (
        f"tr{num_train}_val{num_val}_te{num_test}_trmr{train_mask_ratio}_seed{seed}"
    )
    train_data_path = out_dir / f"{prefix}_tr_data_{split_suffix}.h5"
    val_data_path = out_dir / f"{prefix}_val_data_{split_suffix}.h5"
    test_data_path = out_dir / f"{prefix}_te_data_{split_suffix}.npy"
    output_paths = [train_data_path, val_data_path, test_data_path]
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
    print("\nWriting training data...")
    data_shape = (time_steps, num_depths, height, width, 2)
    chunk_shape = (1, time_steps, num_depths, height, width, 2)
    with h5py.File(train_data_path, "w") as train_h5:
        train_ds = train_h5.create_dataset(
            "data", shape=(num_train, *data_shape), dtype=np.float32, chunks=chunk_shape
        )
        train_ds[:] = train_data
    print("Writing validation data...")
    with h5py.File(val_data_path, "w") as val_h5:
        val_ds = val_h5.create_dataset(
            "data", shape=(num_val, *data_shape), dtype=np.float32, chunks=chunk_shape
        )
        val_ds[:] = val_data
    print("Writing test data...")
    np.save(test_data_path, {"data": test_data})
    u_ind_uni = np.linspace(0, height - 1, height).astype(np.float32) / (height - 1)
    v_ind_uni = np.linspace(0, width - 1, width).astype(np.float32) / (width - 1)
    t_ind_uni = np.linspace(0, time_steps - 1, time_steps).astype(np.float32) / (
        time_steps - 1
    )
    d_ind_uni = np.linspace(0, num_depths - 1, num_depths).astype(np.float32) / (
        num_depths - 1
    )
    lat_coords = np.asarray(metadata["lat_values"], dtype=np.float32)
    lon_coords = np.asarray(metadata["lon_values"], dtype=np.float32)
    t_coords_physical = np.arange(time_steps, dtype=np.float32)
    depth_coords = np.array(metadata["depth_values"], dtype=np.float32)
    train_mask_rng = np.random.RandomState(seed)
    mask_tr = make_per_sequence_time_consistent_mask(
        train_mask_rng, num_train, time_steps, height, width, train_mask_ratio
    )
    indices_tr = np.arange(num_train)
    indices_val = np.arange(num_train, num_train + num_val)
    indices_te = np.arange(num_train + num_val, num_samples_actual)
    common_meta = {
        "u_ind_uni": u_ind_uni,
        "v_ind_uni": v_ind_uni,
        "t_ind_uni": t_ind_uni,
        "d_ind_uni": d_ind_uni,
        "u_coords_physical": lat_coords,
        "v_coords_physical": lon_coords,
        "t_coords_physical": t_coords_physical,
        "d_coords_physical": depth_coords,
        "lat_range": metadata["lat_range"],
        "lon_range": metadata["lon_range"],
        "depth_values": metadata["depth_values"],
        "num_depths": metadata["num_depths"],
        "dx": metadata["dx"],
        "dy": metadata["dy"],
        "split_ratios": np.asarray(split_ratios, dtype=np.float32),
        "seed": seed,
        "indices_tr": indices_tr,
        "indices_val": indices_val,
        "indices_te": indices_te,
        "train_mask_ratio": train_mask_ratio,
        "consistent_points_across_time": True,
        "source_dir": str(data_dir),
        "grid_size": int(height),
        "time_steps": int(time_steps),
        "days_per_sample": int(days_per_sample),
        "full_sample_count": int(num_samples_actual),
        "used_sample_count": int(num_samples_actual),
        "field_name": "ocean_velocity",
        "field_mode": "uv",
        "num_channels": 2,
        "channel_order": np.asarray(["u", "v"], dtype=object),
        "depth_indices": np.arange(num_depths, dtype=np.int64),
        "source_files_used": metadata["source_files_used"],
    }
    manifest_index = {}
    for eval_mask_ratio in eval_mask_ratios:
        ratio_tag = format_ratio_tag(eval_mask_ratio)
        eval_seed = build_eval_seed(seed, eval_mask_ratio)
        eval_rng = np.random.RandomState(eval_seed)
        mask_val = make_per_sequence_time_consistent_mask(
            eval_rng, num_val, time_steps, height, width, eval_mask_ratio
        )
        mask_te = make_per_sequence_time_consistent_mask(
            eval_rng, num_test, time_steps, height, width, eval_mask_ratio
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
        records = metadata["sample_records"]
        train_records, val_records, test_records = (
            records[:num_train],
            records[num_train : num_train + num_val],
            records[num_train + num_val :],
        )
        np.save(
            train_meta_path,
            {
                "data": {
                    **meta_common,
                    "mask_tr": mask_tr,
                    "sample_records": train_records,
                }
            },
        )
        np.save(
            val_meta_path,
            {
                "data": {
                    **meta_common,
                    "mask_val": mask_val,
                    "sample_records": val_records,
                }
            },
        )
        np.save(
            test_meta_path,
            {
                "data": {
                    **meta_common,
                    "mask_te": mask_te,
                    "sample_records": test_records,
                }
            },
        )
        manifest = {
            "data_name": prefix,
            "dim": "3d",
            "split_ratios": list(map(float, split_ratios)),
            "seed": seed,
            "grid_size": int(height),
            "num_samples": int(num_samples_actual),
            "time_steps": int(time_steps),
            "days_per_sample": int(days_per_sample),
            "num_depths": int(num_depths),
            "depth_values": metadata["depth_values"],
            "depth_indices": list(range(num_depths)),
            "source_product": "GLOBAL_MULTIYEAR_PHY_001_030",
            "source_files_used": metadata["source_files_used"],
            "sample_records": records,
            "sequence_rule": "nonoverlapping within-file blocks; trailing records discarded; no cross-file concatenation",
            "field_mode": "uv",
            "num_channels": 2,
            "channel_order": ["u", "v"],
            "train_mask_ratio": float(train_mask_ratio),
            "eval_mask_ratio": float(eval_mask_ratio),
            "consistent_points_across_time": True,
            "source_dir": str(data_dir),
            "lat_range": metadata["lat_range"],
            "lon_range": metadata["lon_range"],
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
        for split, selected_records in zip(
            ("train", "val", "test"), (train_records, val_records, test_records)
        ):
            manifest["splits"][split]["sample_indices"] = [
                r["sample_index"] for r in selected_records
            ]
            manifest["splits"][split]["start_date"] = (
                selected_records[0]["start_date"] if selected_records else None
            )
            manifest["splits"][split]["end_date"] = (
                selected_records[-1]["end_date"] if selected_records else None
            )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        manifest_index[ratio_tag] = str(manifest_path)
    print(f"\nDone! train={num_train} val={num_val} test={num_test}")
    for ratio_tag, manifest_path in manifest_index.items():
        print(f"evmr={ratio_tag} manifest={manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process multi-year ocean data into 100 samples with 8:1:1 split."
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="Directory containing NetCDF files"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--days_per_sample", type=int, default=90, help="Days per sample"
    )
    parser.add_argument(
        "--lat_min", type=float, default=DEFAULT_LAT_RANGE[0], help="Minimum latitude"
    )
    parser.add_argument(
        "--lat_max", type=float, default=DEFAULT_LAT_RANGE[1], help="Maximum latitude"
    )
    parser.add_argument(
        "--lon_min", type=float, default=DEFAULT_LON_RANGE[0], help="Minimum longitude"
    )
    parser.add_argument(
        "--lon_max", type=float, default=DEFAULT_LON_RANGE[1], help="Maximum longitude"
    )
    parser.add_argument(
        "--num_depths",
        type=int,
        default=32,
        help="Take the first depth layers; 32 validates released depth coordinates",
    )
    parser.add_argument(
        "--grid_height", type=int, default=64, help="Target grid height"
    )
    parser.add_argument("--grid_width", type=int, default=64, help="Target grid width")
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        help="Train/val/test split ratios",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--train_mask_ratio", type=float, default=0.15, help="Training mask ratio"
    )
    parser.add_argument(
        "--eval_mask_ratios",
        type=float,
        nargs="+",
        default=[0.15, 0.1, 0.05, 0.03, 0.01],
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for reading",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    args = parser.parse_args()
    process_multi_year_data(
        data_dir=args.data_dir,
        out_dir=args.output_dir,
        num_samples=args.num_samples,
        days_per_sample=args.days_per_sample,
        lat_range=(args.lat_min, args.lat_max),
        lon_range=(args.lon_min, args.lon_max),
        target_size=(args.grid_height, args.grid_width),
        split_ratios=tuple(args.split_ratios),
        seed=args.seed,
        train_mask_ratio=args.train_mask_ratio,
        eval_mask_ratios=tuple(args.eval_mask_ratios),
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        num_depths=args.num_depths,
    )


if __name__ == "__main__":
    main()
