"""Dataset split and auxiliary metadata helpers."""

import json
import os
from pathlib import Path


def _resolve_manifest_file_path(path_value, data_manifest_path):
    path = Path(str(path_value)).expanduser()
    candidates = [path]
    remap_from = str(os.environ.get("PHYSFTM_PATH_REMAP_FROM", "")).strip()
    remap_to = str(os.environ.get("PHYSFTM_PATH_REMAP_TO", "")).strip()
    if remap_from and remap_to:
        raw_text = str(path)
        if raw_text == remap_from or raw_text.startswith(remap_from.rstrip("/") + "/"):
            candidates.append(
                Path(remap_to.rstrip("/") + raw_text[len(remap_from.rstrip("/")) :])
            )
    data_root = str(os.environ.get("PHYSFTM_DATA_ROOT", "")).strip()
    marker = "/data/raw_data/"
    raw_text = str(path)
    marker_index = raw_text.find(marker)
    if data_root and marker_index >= 0:
        candidates.append(
            Path(data_root.rstrip("/") + "/" + raw_text[marker_index + len(marker) :])
        )
    expanded = []
    for candidate in candidates:
        expanded.append(candidate)
        expanded.extend(
            [
                Path(
                    str(candidate).replace("/time_varing_mask/", "/time_varying_mask/")
                ),
                Path(
                    str(candidate).replace("/time_varying_mask/", "/time_varing_mask/")
                ),
            ]
        )
    for candidate in expanded:
        if candidate.exists():
            if str(candidate) != str(path):
                print(f"[manifest] remapped path: {path} -> {candidate}")
            return str(candidate)
    sibling = Path(data_manifest_path).expanduser().resolve().parent / path.name
    if sibling.exists():
        print(f"[manifest] using existing sibling for stale path: {path} -> {sibling}")
        return str(sibling)
    return str(path)


def resolve_split_paths(
    data_manifest_path=None, split="train", data_path=None, metadata_path=None
):
    if data_manifest_path:
        with open(data_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        split_info = manifest["splits"][split]
        return (
            _resolve_manifest_file_path(split_info["data_path"], data_manifest_path),
            _resolve_manifest_file_path(
                split_info["metadata_path"], data_manifest_path
            ),
            manifest,
        )
    return (data_path, metadata_path, None)
