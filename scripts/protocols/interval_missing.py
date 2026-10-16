"""Generate temporal interval-missing metadata and manifests.

For every selected temporal-consistent spatial mask, each sequence receives one
random contiguous missing interval covering 20% of its time steps. The interval
is sampled strictly between the first and last frame, and all spatial points
are removed during that interval. Data files are not copied or modified;
generated manifests continue to reference the source train/validation/test
data and point to the generated metadata.

Example::

    python scripts/scripts/protocols/interval_missing.py
"""

from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any
import numpy as np

DEFAULT_DATASET_DIRS = {
    "AC": Path("data/processed/ac2d"),
    "NS2D": Path("data/processed/ns2d"),
    "Heat3D": Path("data/processed/heat3d"),
    "Ocean": Path("data/processed/ocean"),
}
MASK_KEYS = {"train": "mask_tr", "val": "mask_val", "test": "mask_te"}
TARGET_RATIOS = (0.01, 0.03, 0.15)
INTERVAL_FRACTION = 0.2


def ratio_tag(value: float) -> str:
    return format(float(value), "g")


def load_metadata(path: Path) -> dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    if not isinstance(obj, np.ndarray) or obj.dtype != object or obj.shape != ():
        raise ValueError(f"Unexpected metadata format: {path}")
    value = obj.item()
    if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
        raise ValueError(f"Metadata must contain a data dict: {path}")
    return value


def save_metadata_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_path(manifest_path: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()
    candidate = manifest_path.parent / path.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Manifest path does not exist: {path_text}")


def interval_seed(
    base_seed: int, ratio: float, split_index: int, seed_offset: int
) -> int:
    """Stable seed scheme kept in the manifest for reproducibility."""
    ratio_part = int(round(float(ratio) * 10000.0))
    return (
        int(base_seed) * 1000000
        + int(seed_offset) * 10000
        + ratio_part * 10
        + int(split_index)
    )


def interval_length(time_steps: int, fraction: float) -> int:
    if time_steps < 3:
        raise ValueError(f"At least three time steps are required, got {time_steps}")
    length = max(1, int(round(float(time_steps) * float(fraction))))
    if length > time_steps - 2:
        raise ValueError(
            f"Interval length {length} leaves no protected first/last frame for T={time_steps}"
        )
    return length


def apply_interval_missing(
    base_mask: np.ndarray, seed: int, fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(base_mask)
    if mask.ndim not in (4, 5):
        raise ValueError(
            f"Expected mask shape (N,T,...) or (N,T,...,1), got {mask.shape}"
        )
    num_samples, time_steps = (int(mask.shape[0]), int(mask.shape[1]))
    length = interval_length(time_steps, fraction)
    start_low = 1
    start_high = time_steps - length - 1
    if start_high < start_low:
        raise ValueError(f"No legal interval start for T={time_steps}, length={length}")
    rng = np.random.RandomState(seed)
    starts = rng.randint(start_low, start_high + 1, size=num_samples).astype(np.int64)
    ends = starts + length
    output = np.array(mask, dtype=np.uint8, copy=True)
    for sample_index, (start, end) in enumerate(zip(starts, ends)):
        output[sample_index, int(start) : int(end)] = 0
    return (output, starts, ends)


def discover_manifests(dataset_dir: Path, ratios: set[str]) -> list[Path]:
    selected = []
    for path in sorted(dataset_dir.glob("*_temporal_consistent_mask_manifest_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if ratio_tag(float(manifest.get("eval_mask_ratio"))) in ratios:
            selected.append(path)
    return selected


def process_manifest(
    manifest_path: Path,
    fraction: float,
    seed_offset: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ratio = float(manifest["eval_mask_ratio"])
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or any(
        (split not in splits for split in MASK_KEYS)
    ):
        raise ValueError(f"Manifest missing train/val/test splits: {manifest_path}")
    protocol_dir = manifest_path.parent / "temporal_interval_missing"
    source_metadata = {
        split: resolve_path(manifest_path, splits[split]["metadata_path"])
        for split in MASK_KEYS
    }
    output_manifest = protocol_dir / manifest_path.name.replace(
        "_temporal_consistent_mask_", "_temporal_interval_missing_"
    )
    output_metadata = {
        split: protocol_dir
        / source_metadata[split].name.replace(
            "_temporal_consistent_mask_", "_temporal_interval_missing_"
        )
        for split in MASK_KEYS
    }
    outputs = [output_manifest, *output_metadata.values()]
    existing = [path for path in outputs if path.exists()]
    if existing and (not overwrite) and (not dry_run):
        raise FileExistsError(
            "Outputs already exist; use --overwrite:\n" + "\n".join(map(str, existing))
        )
    print(f"[{manifest_path.parent.name}] ratio={ratio_tag(ratio)}")
    for path in outputs:
        print(f"  {path}")
    if dry_run:
        return
    generated: dict[str, dict[str, Any]] = {}
    base_seed = int(manifest.get("seed", 0))
    for split_index, split in enumerate(("train", "val", "test")):
        metadata = load_metadata(source_metadata[split])
        inner = metadata["data"]
        mask_key = MASK_KEYS[split]
        if mask_key not in inner:
            raise KeyError(f"{source_metadata[split]} missing {mask_key}")
        old_mask = np.asarray(inner[mask_key])
        seed = interval_seed(base_seed, ratio, split_index, seed_offset)
        new_mask, starts, ends = apply_interval_missing(old_mask, seed, fraction)
        inner[mask_key] = new_mask
        inner["consistent_points_across_time"] = False
        inner["spatial_mask_consistent_across_time"] = True
        inner["temporal_protocol"] = "temporal_interval_missing"
        inner["temporal_interval_missing"] = True
        inner["temporal_interval_missing_fraction"] = float(fraction)
        inner["temporal_interval_missing_length"] = int(ends[0] - starts[0])
        inner["temporal_interval_missing_start"] = starts
        inner["temporal_interval_missing_end"] = ends
        inner["temporal_interval_missing_end_inclusive"] = False
        inner["temporal_interval_missing_protected_frames"] = [
            0,
            int(old_mask.shape[1] - 1),
        ]
        inner["temporal_interval_missing_seed"] = int(seed)
        inner["source_metadata_path"] = str(source_metadata[split].resolve())
        save_metadata_atomic(output_metadata[split], metadata)
        generated[split] = {
            "seed": int(seed),
            "num_samples": int(old_mask.shape[0]),
            "time_steps": int(old_mask.shape[1]),
            "interval_length": int(ends[0] - starts[0]),
            "start_min": int(starts.min()),
            "start_max": int(starts.max()),
            "end_min": int(ends.min()),
            "end_max": int(ends.max()),
        }
    protocol_manifest = copy.deepcopy(manifest)
    protocol_manifest["data_name"] = str(
        protocol_manifest.get("data_name", manifest_path.stem)
    ).replace("temporal_consistent_mask", "temporal_interval_missing", 1)
    protocol_manifest["consistent_points_across_time"] = False
    protocol_manifest["spatial_mask_consistent_across_time"] = True
    protocol_manifest["temporal_protocol"] = "temporal_interval_missing"
    protocol_manifest["temporal_interval_missing"] = True
    protocol_manifest["temporal_interval_missing_fraction"] = float(fraction)
    protocol_manifest["temporal_interval_missing_seed_scheme"] = (
        "base_seed*1000000 + seed_offset*10000 + round(eval_mask_ratio*10000)*10 + split_index"
    )
    protocol_manifest["temporal_interval_missing_end_inclusive"] = False
    protocol_manifest["temporal_interval_missing_protected_frames"] = [
        0,
        int(manifest["time_steps"] - 1),
    ]
    protocol_manifest["source_manifest_path"] = str(manifest_path.resolve())
    protocol_manifest["splits"] = copy.deepcopy(splits)
    for split in MASK_KEYS:
        protocol_manifest["splits"][split]["metadata_path"] = str(
            output_metadata[split].resolve()
        )
    protocol_manifest["temporal_interval_missing_summary"] = generated
    save_json_atomic(output_manifest, protocol_manifest)
    print(f"wrote {output_manifest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=tuple(DEFAULT_DATASET_DIRS),
        default=tuple(DEFAULT_DATASET_DIRS),
    )
    parser.add_argument("--ac-dir", type=Path, default=DEFAULT_DATASET_DIRS["AC"])
    parser.add_argument("--ns2d-dir", type=Path, default=DEFAULT_DATASET_DIRS["NS2D"])
    parser.add_argument(
        "--heat3d-dir", type=Path, default=DEFAULT_DATASET_DIRS["Heat3D"]
    )
    parser.add_argument("--ocean-dir", type=Path, default=DEFAULT_DATASET_DIRS["Ocean"])
    parser.add_argument("--ratios", type=float, nargs="+", default=TARGET_RATIOS)
    parser.add_argument("--interval-fraction", type=float, default=INTERVAL_FRACTION)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < float(args.interval_fraction) < 1.0:
        raise ValueError(
            f"interval-fraction must be in (0,1), got {args.interval_fraction}"
        )
    directories = {
        "AC": args.ac_dir,
        "NS2D": args.ns2d_dir,
        "Heat3D": args.heat3d_dir,
        "Ocean": args.ocean_dir,
    }
    ratios = {ratio_tag(value) for value in args.ratios}
    for dataset in args.dataset:
        manifests = discover_manifests(directories[dataset], ratios)
        if not manifests:
            raise FileNotFoundError(
                f"No selected manifests under {directories[dataset]}"
            )
        for manifest_path in manifests:
            process_manifest(
                manifest_path=manifest_path,
                fraction=float(args.interval_fraction),
                seed_offset=int(args.seed_offset),
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )


if __name__ == "__main__":
    main()
