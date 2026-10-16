from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import scipy.io as sio
import torch
import h5py
from physftm.data.axes import ordered_spatial_values, spatial_axis_order_from_manifest
from physftm.rkhs.basis import RKHSFunctionalTucker2D, RKHSFunctionalTucker3D

RKHS_ARTIFACT_VERSION = 3
SUPPORTED_RKHS_ARTIFACT_VERSIONS = (2, 3)


def inverse_normalize_tensor(
    field_t: torch.Tensor, norm_stats: Dict[str, object]
) -> torch.Tensor:

    def _prepare_stat_for_field(stat):
        arr = torch.as_tensor(stat, dtype=field_t.dtype, device=field_t.device)
        if arr.ndim == 0:
            return arr.reshape((1,) * field_t.ndim)
        if arr.ndim == 1 and arr.shape[0] == field_t.shape[-1]:
            return arr.reshape((1,) * (field_t.ndim - 1) + (field_t.shape[-1],))
        while arr.ndim < field_t.ndim:
            arr = arr.unsqueeze(0)
        return arr

    if bool(norm_stats.get("map_nonnegative", False)):
        shift = _prepare_stat_for_field(norm_stats.get("nonnegative_shift", 0.0))
        field_t = field_t - shift
    if bool(norm_stats.get("map_signed", False)):
        half = torch.clamp(
            _prepare_stat_for_field(norm_stats.get("signed_half_range", 1.0)), min=1e-08
        )
        center = _prepare_stat_for_field(norm_stats.get("signed_center", 0.0))
        field_t = field_t * half + center
    norm_mode = norm_stats.get("norm_mode", "none")
    if norm_mode in (None, "none"):
        return field_t
    if norm_mode in ("zscore", "channelwise_zscore"):
        mean = _prepare_stat_for_field(norm_stats.get("data_mean", 0.0))
        std = torch.clamp(
            _prepare_stat_for_field(norm_stats.get("data_std", 1.0)), min=1e-08
        )
        return field_t * std + mean
    raise ValueError(f"Unsupported norm_mode: {norm_mode}")


def load_ftm_manifest(ftm_manifest_path: str) -> Dict[str, object]:
    with open(ftm_manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_current_rkhs_artifact(ftm_manifest: Dict[str, object]) -> None:
    version = ftm_manifest.get("artifact_version")
    try:
        version_int = int(version or 0)
    except (TypeError, ValueError):
        version_int = 0
    if version_int not in SUPPORTED_RKHS_ARTIFACT_VERSIONS:
        raise ValueError(
            f"Unsupported RKHS artifact schema version {version!r}; supported versions are {SUPPORTED_RKHS_ARTIFACT_VERSIONS}."
        )


def _resolve_manifest_path(
    ftm_manifest: Dict[str, object], manifest_path: str | Path, key: str
) -> Path:
    """Resolve current-schema artifact paths relative to the manifest directory first."""
    manifest_dir = Path(manifest_path).resolve().parent
    paths = ftm_manifest.get("paths", {})
    raw_value = paths.get(key, ftm_manifest.get(key))
    if raw_value is None:
        raise KeyError(f"Manifest is missing artifact path key: {key}")
    raw_path = Path(str(raw_value))
    if raw_path.is_absolute():
        return raw_path
    candidates = [
        (manifest_dir / raw_path).resolve(),
        (Path.cwd() / raw_path).resolve(),
        (manifest_dir / raw_path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _manifest_norm_stats(ftm_manifest: Dict[str, object]) -> Dict[str, object]:
    normalization = ftm_manifest.get("normalization", {})
    if not normalization:
        raise ValueError(
            "RKHS current artifact manifest is missing normalization stats."
        )
    return {
        "norm_mode": normalization.get("norm_mode", "none"),
        "data_mean": normalization.get("data_mean", 0.0),
        "data_std": normalization.get("data_std", 1.0),
        "data_min": normalization.get("data_min", 0.0),
        "data_max": normalization.get("data_max", 1.0),
        "map_nonnegative": bool(normalization.get("map_nonnegative", False)),
        "nonnegative_shift": normalization.get("nonnegative_shift", 0.0),
        "map_signed": bool(normalization.get("map_signed", False)),
        "signed_center": normalization.get("signed_center", 0.0),
        "signed_half_range": normalization.get("signed_half_range", 1.0),
    }


def _load_core_payload_current(
    core_path: str, ftm_manifest: Dict[str, object]
) -> Dict[str, object]:
    validate_current_rkhs_artifact(ftm_manifest)
    suffix = Path(core_path).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        core_cfg = ftm_manifest.get("core", {})
        dataset_names = (
            core_cfg.get("datasets", ["core"])
            if isinstance(core_cfg, dict)
            else ["core"]
        )
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        with h5py.File(core_path, "r") as handle:
            dataset_name = next(
                (name for name in dataset_names if name in handle), None
            )
            if dataset_name is None:
                raise ValueError(
                    f"RKHS current artifact core file is missing datasets {dataset_names}: {core_path}"
                )
            core_total = np.asarray(handle[dataset_name], dtype=np.float32)
    else:
        mat = sio.loadmat(core_path, variable_names=["core"])
        if "core" not in mat:
            raise ValueError(
                f"RKHS current artifact core file is missing 'core': {core_path}"
            )
        core_total = np.asarray(mat["core"], dtype=np.float32)
    model_cfg = ftm_manifest.get("model", {})
    shared_basis_multichannel = bool(
        model_cfg.get(
            "shared_basis_multichannel",
            ftm_manifest.get("shared_basis_multichannel", False),
        )
    )
    num_channels = int(
        model_cfg.get("num_channels", ftm_manifest.get("num_channels", 1))
    )
    return {
        "core": core_total,
        "core_spatial": core_total,
        "norm_stats": _manifest_norm_stats(ftm_manifest),
        "shared_basis_multichannel": shared_basis_multichannel,
        "num_channels": num_channels,
    }


def load_core_payload(
    core_path: str,
    *,
    ftm_manifest: Optional[Dict[str, object]] = None,
    load_core_total: bool = True,
    load_core_spectral: bool = True,
) -> Dict[str, object]:
    del load_core_total
    del load_core_spectral
    if ftm_manifest is None:
        raise ValueError("load_core_payload requires a current RKHS artifact manifest.")
    return _load_core_payload_current(core_path, ftm_manifest)


def _coords_from_manifest(ftm_manifest: Dict[str, object], device: torch.device):
    coordinates = ftm_manifest.get("coordinates", {})
    required = ["t", "u", "v"]
    missing = [key for key in required if key not in coordinates]
    if missing:
        raise ValueError(
            f"RKHS current artifact manifest is missing coordinate fields: {missing}"
        )
    t_ind = torch.as_tensor(
        coordinates["t"], dtype=torch.float32, device=device
    ).reshape(-1)
    u_ind = torch.as_tensor(
        coordinates["u"], dtype=torch.float32, device=device
    ).reshape(-1)
    v_ind = torch.as_tensor(
        coordinates["v"], dtype=torch.float32, device=device
    ).reshape(-1)
    w_ind = (
        None
        if "w" not in coordinates
        else torch.as_tensor(
            coordinates["w"], dtype=torch.float32, device=device
        ).reshape(-1)
    )
    return (t_ind, u_ind, v_ind, w_ind)


def _inducing_coords_from_manifest(
    ftm_manifest, device, *, core_parameterization, full_coords, spatial_axis_order
):
    inducing = ftm_manifest.get("inducing_coordinates")
    if not inducing:
        return full_coords[1:]
    required = tuple(spatial_axis_order)
    missing = [key for key in required if key not in inducing]
    if missing:
        raise ValueError(
            f"RKHS artifact manifest is missing inducing coordinate fields: {missing}"
        )
    return tuple(
        (
            torch.as_tensor(inducing[key], dtype=torch.float32, device=device).reshape(
                -1
            )
            for key in required
        )
    )


def _lengthscales_from_manifest(basis_cfg, axes):
    values = basis_cfg.get("effective_lengthscales", {})
    if not isinstance(values, dict):
        return None
    out = tuple(
        (None if values.get(axis) is None else float(values[axis]) for axis in axes)
    )
    return None if all((value is None for value in out)) else out


def _build_basis_from_manifest(ftm_manifest: Dict[str, object], device: torch.device):
    validate_current_rkhs_artifact(ftm_manifest)
    basis_cfg = ftm_manifest.get("basis", {})
    if str(ftm_manifest.get("core_parameterization", "per_time")).lower() != "per_time":
        raise ValueError("This release supports per-time spatial RKHS artifacts")
    if str(basis_cfg.get("kernel_type", "matern32")).lower() != "matern32":
        raise ValueError("This release supports Matérn-3/2 spatial kernels")
    if not basis_cfg:
        raise ValueError(
            "RKHS current artifact manifest is missing basis construction config."
        )
    t_ind, u_ind, v_ind, w_ind = _coords_from_manifest(ftm_manifest, device)
    is_3d = w_ind is not None and int(w_ind.numel()) > 1 and ("per_time" == "per_time")
    spatial_axis_order = spatial_axis_order_from_manifest(
        ftm_manifest, spatial_dim=3 if is_3d else 2
    )
    spatial_coords = ordered_spatial_values(
        spatial_axis_order, u=u_ind, v=v_ind, w=w_ind if is_3d else None
    )
    full_coords = (t_ind, *spatial_coords)
    rank = tuple((int(x) for x in basis_cfg.get("rank", ftm_manifest.get("rank", []))))
    if len(rank) < 2:
        raise ValueError(
            f"Per-time RKHS artifact needs at least spatial rank length 2, got {rank}"
        )
    per_time_full_coords = full_coords
    inducing_points = _inducing_coords_from_manifest(
        ftm_manifest,
        device,
        core_parameterization="per_time",
        full_coords=per_time_full_coords,
        spatial_axis_order=spatial_axis_order,
    )
    if is_3d:
        if len(rank) != 3:
            raise ValueError(
                f"3D per-time RKHS artifact needs rank length 3, got {rank}"
            )
        inducing_points = _inducing_coords_from_manifest(
            ftm_manifest,
            device,
            core_parameterization="per_time",
            full_coords=per_time_full_coords,
            spatial_axis_order=spatial_axis_order,
        )
        basis_function = RKHSFunctionalTucker3D(
            ranks=rank[:3],
            inducing_points=inducing_points,
            lengthscales=_lengthscales_from_manifest(basis_cfg, spatial_axis_order),
            kernel_type=str(basis_cfg.get("kernel_type", "rbf")),
            coef_scale=float(basis_cfg.get("coef_scale", 0.01)),
        ).to(device)
        ind_input = spatial_coords
    else:
        basis_function = RKHSFunctionalTucker2D(
            ranks=rank[:2],
            inducing_points=inducing_points,
            lengthscales=_lengthscales_from_manifest(basis_cfg, ("u", "v")),
            kernel_type=str(basis_cfg.get("kernel_type", "rbf")),
            coef_scale=float(basis_cfg.get("coef_scale", 0.01)),
        ).to(device)
        ind_input = (u_ind, v_ind)
    meta = {
        "t_ind_uni": t_ind.detach().cpu().numpy(),
        "u_ind_uni": u_ind.detach().cpu().numpy(),
        "v_ind_uni": v_ind.detach().cpu().numpy(),
        "w_ind_uni": None if w_ind is None else w_ind.detach().cpu().numpy(),
        "spatial_axis_order": spatial_axis_order,
        "inducing_points": tuple(
            (coord.detach().cpu().numpy() for coord in inducing_points)
        ),
    }
    return (basis_function, ind_input, meta)


def _load_basis_and_coords_current(ftm_manifest_path: str, device: torch.device):
    ftm_manifest = load_ftm_manifest(ftm_manifest_path)
    validate_current_rkhs_artifact(ftm_manifest)
    basis_path = _resolve_manifest_path(ftm_manifest, ftm_manifest_path, "basis_path")
    if not basis_path.exists():
        raise FileNotFoundError(
            f"RKHS current artifact basis file does not exist: {basis_path}"
        )
    basis_function, ind_input, meta = _build_basis_from_manifest(ftm_manifest, device)
    try:
        payload = torch.load(basis_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(basis_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(
            f"RKHS current artifact basis payload must be a dict: {basis_path}"
        )
    manifest_version = int(ftm_manifest.get("artifact_version", 0))
    payload_version = int(payload.get("format_version", 0))
    if (
        payload_version not in SUPPORTED_RKHS_ARTIFACT_VERSIONS
        or payload_version != manifest_version
    ):
        raise ValueError(
            f"Unsupported RKHS basis payload version in {basis_path}: {payload.get('format_version')}"
        )
    if payload.get("artifact_type") != "rkhs_basis_state":
        raise ValueError(
            f"Unsupported RKHS basis artifact_type in {basis_path}: {payload.get('artifact_type')}"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"RKHS current artifact basis payload missing state_dict: {basis_path}"
        )
    basis_function.load_state_dict(state_dict)
    basis_function.eval()
    ftm_manifest["basis_path"] = str(basis_path)
    try:
        ftm_manifest["core_path"] = str(
            _resolve_manifest_path(ftm_manifest, ftm_manifest_path, "core_path")
        )
    except KeyError:
        pass
    return (ftm_manifest, basis_function, ind_input, meta)


def load_basis_and_coords(ftm_manifest_path: str, device: torch.device):
    return _load_basis_and_coords_current(ftm_manifest_path, device)
