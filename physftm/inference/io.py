import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import torch
from physftm.artifacts.loading import load_ftm_manifest
from physftm.data.loading import align_3d_mask_to_data

REPO_ROOT = str(Path(__file__).resolve().parents[2])
DEFAULT_RESULTS_ROOT = str(Path(REPO_ROOT) / "results" / "discrete")


@dataclass
class RKHSInferenceResult:
    results_dir: str
    predictions_path: str
    metrics_path: str
    inference_times_path: str
    metrics: dict


def _resolve_checkpoint_paths(args):
    if args.ftm_manifest_path:
        ftm_manifest = load_ftm_manifest(args.ftm_manifest_path)
        basis_path = _resolve_manifest_artifact_path(
            ftm_manifest["basis_path"], args.ftm_manifest_path
        )
        core_path = _resolve_manifest_artifact_path(
            ftm_manifest["core_path"], args.ftm_manifest_path
        )
        ftm_manifest["basis_path"] = basis_path
        ftm_manifest["core_path"] = core_path
        return (ftm_manifest, basis_path, core_path)
    raise ValueError(
        "Strict RKHS artifact loading requires --ftm_manifest_path for the current schema."
    )


def _resolve_manifest_artifact_path(path_text, manifest_path):
    path_text = str(path_text)
    manifest_dir = Path(manifest_path).resolve().parent
    script_dir = Path(__file__).resolve().parent
    raw_path = Path(path_text)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                (Path.cwd() / raw_path).resolve(),
                (script_dir / raw_path).resolve(),
                (manifest_dir / raw_path.name).resolve(),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            if str(candidate) != path_text:
                print(
                    f"resolved missing manifest artifact path: {path_text} -> {candidate}"
                )
            return str(candidate)
    return path_text


def _build_results_dir(path, prefix="basis_split_reconstruction"):
    if path:
        out_dir = path
    else:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        safe_prefix = "".join(
            (
                ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
                for ch in str(prefix).strip()
            )
        )
        safe_prefix = safe_prefix or "basis_split_reconstruction"
        out_dir = os.path.join(DEFAULT_RESULTS_ROOT, f"{safe_prefix}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    return out_dir


def _ensure_5d(data_np, spatial_dim=None):
    data_t = torch.as_tensor(np.asarray(data_np, dtype=np.float32))
    if spatial_dim == 3:
        if data_t.dim() == 5:
            data_t = data_t.unsqueeze(-1)
    elif spatial_dim == 2:
        if data_t.dim() == 4:
            data_t = data_t.unsqueeze(-1)
    elif data_t.dim() == 4:
        data_t = data_t.unsqueeze(-1)
    if data_t.dim() not in (5, 6):
        raise ValueError(
            f"Expected data shape (N,T,H,W[,C]) or (N,T,D,H,W[,C]), got {tuple(data_t.shape)}"
        )
    return data_t


def _crop_time_axis_to_core_if_requested(data, mask_np, ind_uni, core_train_np, mode):
    mode = str(mode or "none").strip().lower()
    if mode == "none":
        return (data, mask_np, ind_uni, None)
    if mode != "front":
        raise ValueError(f"Unsupported auto_crop_time_to_core={mode!r}")
    target_nt = int(core_train_np.shape[1])
    data_nt = int(data.shape[1])
    if data_nt == target_nt:
        return (data, mask_np, ind_uni, None)
    if data_nt < target_nt:
        raise ValueError(
            f"Cannot crop target data with time length {data_nt} up to saved core time length {target_nt}"
        )
    drop_first = data_nt - target_nt
    data = data[:, drop_first:]
    mask_np = np.asarray(mask_np)[:, drop_first:]
    t_ind_uni = np.asarray(ind_uni[3])
    if int(t_ind_uni.shape[0]) == data_nt:
        t_ind_uni = t_ind_uni[drop_first:]
    ind_uni = (ind_uni[0], ind_uni[1], ind_uni[2], t_ind_uni)
    info = {
        "mode": mode,
        "original_time_length": data_nt,
        "cropped_time_length": target_nt,
        "drop_first_frames": drop_first,
    }
    return (data, mask_np, ind_uni, info)


def _infer_channel_names(data_manifest, num_channels):
    names = None
    if isinstance(data_manifest, dict):
        names = data_manifest.get("channel_order")
    if not names:
        if int(num_channels) == 1:
            names = ["field"]
        elif int(num_channels) == 3:
            names = ["u", "v", "b/h"]
        else:
            names = [f"ch{idx}" for idx in range(int(num_channels))]
    names = [str(x) for x in names]
    if len(names) < int(num_channels):
        names = names + [f"ch{idx}" for idx in range(len(names), int(num_channels))]
    return names[: int(num_channels)]


def _load_npy_flexible(data_path):
    try:
        arr = np.load(data_path, mmap_mode="r")
    except ValueError as exc:
        if "memory-mapped" not in str(exc):
            raise
        arr = np.load(data_path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return arr
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        obj = arr.item()
        if isinstance(obj, dict):
            if "data" not in obj:
                raise KeyError(
                    f"Object-array npy at {data_path} does not contain key 'data'"
                )
            return obj["data"]
        if isinstance(obj, np.ndarray):
            return obj
        raise TypeError(
            f"Unsupported object payload type in {data_path}: {type(obj).__name__}"
        )
    return arr


def _load_large_data_flexible(data_path, metadata_path):
    data_path = str(data_path)
    if data_path.endswith(".npy"):
        data_extract = _load_npy_flexible(data_path)
    else:
        with h5py.File(data_path, "r") as f:
            data_extract = f["data"][:]
    d = np.load(metadata_path, allow_pickle=True).item()["data"]
    if "mask_tr" in d:
        mask = d["mask_tr"]
    elif "mask_val" in d:
        mask = d["mask_val"]
    elif "mask_te" in d:
        mask = d["mask_te"]
    else:
        raise KeyError(
            f"No mask found in metadata {metadata_path}; expected one of mask_tr/mask_val/mask_te"
        )
    depth_key = (
        "w_ind_uni" if "w_ind_uni" in d else "d_ind_uni" if "d_ind_uni" in d else None
    )
    if depth_key is not None:
        ind_uni = (d["u_ind_uni"], d["v_ind_uni"], d[depth_key], d["t_ind_uni"])
        data_ref = torch.as_tensor(np.asarray(data_extract))
        if data_ref.dim() == 5:
            data_ref = data_ref.unsqueeze(-1)
        mask = align_3d_mask_to_data(
            torch.as_tensor(np.asarray(mask)), data_ref
        ).numpy()
    else:
        w_ind_uni = np.zeros((1,), dtype=np.float32)
        ind_uni = (d["u_ind_uni"], d["v_ind_uni"], w_ind_uni, d["t_ind_uni"])
        if mask.ndim == 4:
            mask = mask[..., None]
        elif mask.ndim != 5:
            raise ValueError(
                f"Metadata mask must be per-sequence (N,T,H,W[,1]); got {mask.shape} in {metadata_path}"
            )
    return (ind_uni, data_extract, mask)


def _parse_plot_time_indices(text, nt):
    if isinstance(text, (list, tuple, np.ndarray)):
        raw = list(text)
    else:
        raw = [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]
    if not raw:
        return [0, max(nt // 2, 0), max(nt - 1, 0)]
    out = []
    for item in raw:
        idx = int(item)
        idx = min(max(idx, 0), max(nt - 1, 0))
        out.append(idx)
    return sorted(set(out))


def _sync_device_if_needed(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
