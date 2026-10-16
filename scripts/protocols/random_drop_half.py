"""Generate irregular-temporal metadata with random half-frame drops.

For every selected temporal-consistent-mask manifest, this script creates a
``temporal_protocol`` sibling directory.  Interior sample/time frames receive
independent Bernoulli draws with ``p=0.5`` for dropping the complete frame;
the first and last frames are always retained. A dropped frame has no
observed spatial points; a kept frame retains the source spatial mask. Source
data, metadata, and manifests are never modified.

Example::

    python scripts/protocols/random_drop_half.py
"""

from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np

DEFAULT_DATASET_DIRS = {
    "AC": Path("data/processed/ac2d"),
    "NS2D": Path("data/processed/ns2d"),
    "Heat3D": Path("data/processed/heat3d"),
    "Ocean": Path("data/processed/ocean"),
}
MASK_KEYS = {"train": "mask_tr", "val": "mask_val", "test": "mask_te"}
TARGET_RATIOS = (0.01, 0.03, 0.15)
DROP_PROBABILITY = 0.5


def ratio_tag(value: float) -> str:
    return format(float(value), "g")


def load_metadata(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if not isinstance(obj, np.ndarray) or obj.dtype != object or obj.shape != ():
        raise ValueError(f"Unexpected metadata format: {path}")
    value = obj.item()
    if not isinstance(value, dict) or not isinstance(value.get("data"), dict):
        raise ValueError(f"Metadata must contain a data dict: {path}")
    return value


def save_metadata_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def protocol_seed(base_seed: int, ratio: float, split_index: int) -> int:
    """Derive stable, separate seeds for each split and eval ratio."""
    tag = ratio_tag(ratio).replace(".", "").replace("-", "m")
    return int(base_seed) * 100000 + int(tag) * 10 + int(split_index)


def apply_frame_dropout(
    base_mask: np.ndarray, seed: int, drop_probability: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(base_mask)
    if mask.ndim not in (4, 5):
        raise ValueError(
            f"Expected mask shape (N,T,...) or (N,T,...,1), got {mask.shape}"
        )
    if not 0.0 <= drop_probability <= 1.0:
        raise ValueError(f"drop_probability must be in [0,1], got {drop_probability}")
    num_samples, time_steps = (int(mask.shape[0]), int(mask.shape[1]))
    rng = np.random.RandomState(seed)
    frame_keep = rng.rand(num_samples, time_steps) >= float(drop_probability)
    if time_steps > 0:
        frame_keep[:, 0] = True
        frame_keep[:, -1] = True
    frame_gate = frame_keep.reshape((num_samples, time_steps) + (1,) * (mask.ndim - 2))
    output = (mask.astype(np.uint8, copy=False) * frame_gate.astype(np.uint8)).astype(
        np.uint8, copy=False
    )
    return (output, frame_keep.astype(np.uint8, copy=False))


def resolve_path(manifest_path: Path, path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path
    relative = manifest_path.parent / path.name
    if relative.exists():
        return relative
    raise FileNotFoundError(f"Manifest path does not exist: {path_text}")


def protocol_paths(
    manifest_path: Path, protocol_dir: Path
) -> tuple[Path, dict[str, Path]]:
    name = manifest_path.name.replace(
        "_temporal_consistent_mask_", "_random_drop_half_"
    )
    output_manifest = protocol_dir / name
    metadata_paths: dict[str, Path] = {}
    for split, info in ("train", "val", "test"):
        source = Path(info)
        metadata_paths[split] = protocol_dir / source.name.replace(
            "_temporal_consistent_mask_", "_random_drop_half_"
        )
    return (output_manifest, metadata_paths)


def process_manifest(
    manifest_path: Path,
    ratio: float,
    drop_probability: float,
    overwrite: bool,
    dry_run: bool,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_ratio = float(manifest.get("eval_mask_ratio"))
    if ratio_tag(actual_ratio) != ratio_tag(ratio):
        raise ValueError(
            f"Ratio mismatch for {manifest_path}: expected {ratio}, got {actual_ratio}"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or any(
        (split not in splits for split in MASK_KEYS)
    ):
        raise ValueError(f"Manifest missing train/val/test splits: {manifest_path}")
    protocol_dir = manifest_path.parent / "temporal_protocol"
    source_metadata = {
        split: resolve_path(manifest_path, splits[split]["metadata_path"])
        for split in MASK_KEYS
    }
    output_manifest = protocol_dir / manifest_path.name.replace(
        "_temporal_consistent_mask_", "_random_drop_half_"
    )
    output_metadata = {
        split: protocol_dir
        / source_metadata[split].name.replace(
            "_temporal_consistent_mask_", "_random_drop_half_"
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
    base_seed = int(manifest.get("seed", 0))
    generated: dict[str, dict] = {}
    for split_index, split in enumerate(("train", "val", "test")):
        metadata = load_metadata(source_metadata[split])
        inner = metadata["data"]
        mask_key = MASK_KEYS[split]
        if mask_key not in inner:
            raise KeyError(f"{source_metadata[split]} missing {mask_key}")
        seed = protocol_seed(base_seed, ratio, split_index)
        new_mask, frame_keep = apply_frame_dropout(
            np.asarray(inner[mask_key]), seed=seed, drop_probability=drop_probability
        )
        inner[mask_key] = new_mask
        inner["temporal_protocol"] = "random_drop_half"
        inner["temporal_dropout_probability"] = float(drop_probability)
        inner["temporal_dropout_keep_probability"] = float(1.0 - drop_probability)
        inner["temporal_dropout_keep_boundary_frames"] = True
        inner["temporal_dropout_seed"] = int(seed)
        inner["temporal_dropout_frame_keep"] = frame_keep
        inner["source_metadata_path"] = str(source_metadata[split].resolve())
        save_metadata_atomic(output_metadata[split], metadata)
        generated[split] = {
            "seed": seed,
            "num_kept_frames": int(frame_keep.sum()),
            "num_frames": int(frame_keep.size),
        }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    protocol_manifest = copy.deepcopy(manifest)
    protocol_manifest["data_name"] = str(
        protocol_manifest.get("data_name", manifest_path.stem)
    ).replace("temporal_consistent_mask", "random_drop_half", 1)
    protocol_manifest["temporal_protocol"] = "random_drop_half"
    protocol_manifest["temporal_dropout_probability"] = float(drop_probability)
    protocol_manifest["temporal_dropout_keep_probability"] = float(
        1.0 - drop_probability
    )
    protocol_manifest["temporal_dropout_keep_boundary_frames"] = True
    protocol_manifest["temporal_dropout_seed_scheme"] = (
        "seed*100000 + ratio_tag*10 + split_index"
    )
    protocol_manifest["source_manifest_path"] = str(manifest_path.resolve())
    protocol_manifest["splits"] = copy.deepcopy(splits)
    for split in MASK_KEYS:
        protocol_manifest["splits"][split]["metadata_path"] = str(
            output_metadata[split].resolve()
        )
    protocol_manifest["temporal_dropout_summary"] = generated
    with output_manifest.open("w", encoding="utf-8") as handle:
        json.dump(protocol_manifest, handle, indent=2)
    print(f"wrote {output_manifest}")


def discover_manifests(dataset_dir: Path, ratios: set[str]) -> list[Path]:
    manifests = sorted(dataset_dir.glob("*_temporal_consistent_mask_manifest_*.json"))
    selected = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if ratio_tag(float(manifest.get("eval_mask_ratio"))) in ratios:
            selected.append(path)
    return selected


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
    parser.add_argument("--drop-probability", type=float, default=DROP_PROBABILITY)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Add a reproducible offset to protocol seeds.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    directories = {
        "AC": args.ac_dir,
        "NS2D": args.ns2d_dir,
        "Heat3D": args.heat3d_dir,
        "Ocean": args.ocean_dir,
    }
    ratios = {ratio_tag(value) for value in args.ratios}
    for dataset in args.dataset:
        dataset_dir = directories[dataset]
        manifests = discover_manifests(dataset_dir, ratios)
        if not manifests:
            raise FileNotFoundError(f"No selected manifests found under {dataset_dir}")
        for manifest_path in manifests:
            process_manifest(
                manifest_path,
                ratio=float(
                    json.loads(manifest_path.read_text(encoding="utf-8"))[
                        "eval_mask_ratio"
                    ]
                ),
                drop_probability=float(args.drop_probability),
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )


if __name__ == "__main__":
    main()
