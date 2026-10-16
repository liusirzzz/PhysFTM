def _map_time_coords_to_physical(query_times, source_times, source_physical_times):
    query = np.asarray(query_times, dtype=np.float64).reshape(-1)
    source = np.asarray(source_times, dtype=np.float64).reshape(-1)
    physical = np.asarray(source_physical_times, dtype=np.float64).reshape(-1)
    if source.size != physical.size:
        return query.astype(np.float32)
    return np.interp(query, source, physical).astype(np.float32)


from physftm.pde.query_fd import (
    build_dense_field_query_fn_3d,
    compute_heat3d_pde_loss_fd_query,
    _sample_field_with_mode_basis_3d,
)
import copy
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "physftm-matplotlib"))
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from physftm.artifacts.loading import (
    inverse_normalize_tensor,
    load_basis_and_coords,
    load_core_payload,
)
from physftm.data.normalization import apply_saved_normalization
from physftm.data.splits import resolve_split_paths
from physftm.decoding.tucker import decode_spatial_from_precomputed_basis
from physftm.decoding.tucker import (
    decode_from_basis_3d,
    decode_from_basis_3d_shared_channels,
)
from physftm.data.axes import ordered_spatial_values, spatial_axis_order_from_manifest
from physftm.pde.query_fd import (
    _sample_continuous_spatial_points,
    _sample_field_with_mode_basis_2d,
    build_dense_field_query_fn_2d,
    compute_ac_pde_loss_fd_query,
    compute_ns_pde_loss_fd_query,
)
from physftm.pde.training import (
    _build_plot_sample_coords,
    _compute_offgrid_point_residual_terms,
    _restore_active_pde_query_chunk_size,
    _set_active_pde_query_chunk_size,
    pde_channel_order,
)
from physftm.rkhs.core_solvers import (
    ridge_tv_solve_core_2d_from_sparse_mask,
    ridge_tv_solve_core_3d_from_sparse_mask,
)
from physftm.solvers.shared_basis import (
    solve_shared_basis_multichannel_core_2d_from_sparse_mask,
    solve_shared_basis_multichannel_core_3d_from_sparse_mask,
)
from physftm.data.loading import get_mask_batch
from physftm.inference.pde_eval import ns_equation_export_terms
from physftm.inference.temporal_config import (
    _parse_args,
    save_resolved_temporal_inference_config,
)
from physftm.inference.core import (
    _as_core_train_array,
    _check_core_compatible,
    _mean_prior_core_batch,
)
from physftm.inference.io import (
    _ensure_5d,
    _infer_channel_names,
    _load_large_data_flexible,
    _load_npy_flexible,
    _parse_plot_time_indices,
    _resolve_checkpoint_paths,
)
from physftm.inference.metrics import (
    _accumulate_channel_metrics,
    compute_ssim_metrics_batched,
    _finalize_channel_metrics,
    _plot_sample,
)
from physftm.inference.protocol_metrics import (
    accumulate_protocol_rmse,
    protocol_rmse_accumulator,
)

DEFAULT_RESULTS_ROOT = "results/temporal"
TEMPORAL_MODEL_ARCH = "hermite_conditioned_bidirectional_odernn_residual"


def _json_safe(value):
    """Convert nested NumPy containers/scalars to values accepted by json.dump."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode_with_precomputed_basis(
    precomputed_basis,
    core_batch,
    shared_basis_multichannel=False,
    core_parameterization="per_time",
):
    if len(precomputed_basis) == 3 and core_batch.dim() in (5, 6):
        U, V, W = precomputed_basis
        return (
            decode_from_basis_3d_shared_channels(U, V, W, core_batch)
            if shared_basis_multichannel
            else decode_from_basis_3d(U, V, W, core_batch)
        )
    U, V = precomputed_basis[:2]
    return decode_spatial_from_precomputed_basis(
        U,
        V,
        None,
        core_batch,
        shared_basis_multichannel=shared_basis_multichannel,
        core_parameterization="per_time",
    )


def _sync_device_if_needed(device) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class _TeeStdout:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _build_results_dir(path, prefix):
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
        safe_prefix = safe_prefix or "temporal_continuous"
        out_dir = os.path.join(DEFAULT_RESULTS_ROOT, f"{safe_prefix}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots", "fields"), exist_ok=True)
    return out_dir


def _parse_query_times(args, dataset_times):
    dataset_times = np.asarray(dataset_times, dtype=np.float32).reshape(-1)
    if isinstance(args.query_times, (list, tuple, np.ndarray)):
        values = [float(x) for x in args.query_times]
    elif str(args.query_times).strip():
        values = [
            float(x)
            for x in str(args.query_times).replace(";", ",").split(",")
            if str(x).strip()
        ]
    else:
        values = []
    if values:
        values = np.asarray(sorted(values), dtype=np.float32)
        if np.any(np.diff(values) <= 0):
            raise ValueError("temporal.query.times must contain unique values.")
        return values
    if int(args.query_num or 0) > 0:
        start = float(
            dataset_times[0] if args.query_time_start is None else args.query_time_start
        )
        end = float(
            dataset_times[-1] if args.query_time_end is None else args.query_time_end
        )
        return np.linspace(start, end, int(args.query_num), dtype=np.float32)
    return dataset_times.copy()


def _validate_per_time_split(
    ind_uni, data, mask_np, ftm_manifest, split_name, tol=1e-06
):
    coordinates = ftm_manifest.get("coordinates", {})
    spatial_dim = 3 if data.dim() == 6 else 2
    spatial_order = spatial_axis_order_from_manifest(
        ftm_manifest, spatial_dim=spatial_dim
    )
    raw_actual = {
        "u": np.asarray(ind_uni[0], dtype=np.float32).reshape(-1),
        "v": np.asarray(ind_uni[1], dtype=np.float32).reshape(-1),
        "w": np.asarray(ind_uni[2], dtype=np.float32).reshape(-1),
        "t": np.asarray(ind_uni[3], dtype=np.float32).reshape(-1),
    }
    expected = {
        axis: np.asarray(coordinates.get(axis, []), dtype=np.float32).reshape(-1)
        for axis in (*spatial_order, "t")
    }
    actual = {axis: raw_actual[axis] for axis in (*spatial_order, "t")}
    for axis in spatial_order:
        if expected[axis].shape != actual[axis].shape or not np.allclose(
            expected[axis], actual[axis], atol=float(tol), rtol=0.0
        ):
            raise ValueError(
                f"{split_name} {axis}-coordinates do not match the trained RKHS artifact; independent continuous coordinate sets are not supported by this pipeline."
            )
    if expected["t"].size != actual["t"].size:
        raise ValueError(
            f"{split_name} t-coordinate count {actual['t'].size} does not match the trained artifact count {expected['t'].size}."
        )
    expected_shape = (
        int(actual["t"].size),
        *(int(actual[axis].size) for axis in spatial_order),
    )
    if tuple((int(x) for x in data.shape[1 : 2 + spatial_dim])) != expected_shape:
        raise ValueError(
            f"{split_name} data grid {tuple(data.shape[1:2 + spatial_dim])} does not match coordinates {expected_shape}."
        )
    mask = _ensure_mask_for_data(mask_np, spatial_dim=spatial_dim)
    if tuple(mask.shape[: 2 + spatial_dim]) != tuple(data.shape[: 2 + spatial_dim]):
        raise ValueError(
            f"{split_name} mask shape {mask.shape} does not match data shape {tuple(data.shape)}."
        )
    if int(mask.shape[1]) > 1:
        spatial_axes = tuple(range(2, mask.ndim))
        reference = np.max(mask, axis=1, keepdims=True)
        frame_matches_reference = np.all(mask == reference, axis=spatial_axes)
        frame_is_missing = np.all(mask == 0, axis=spatial_axes)
        valid_samples = np.all(frame_matches_reference | frame_is_missing, axis=1)
        if not bool(np.all(valid_samples)):
            bad = np.flatnonzero(~valid_samples)[:8].tolist()
            raise ValueError(
                f"{split_name} has time-varying spatial sensor masks for samples {bad}; per-time temporal reconstruction currently requires each sample's sensor positions to stay fixed over time (fully missing frames and different masks between samples are allowed)."
            )


def _detect_temporal_checkpoint_arch(payload):
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Temporal checkpoint is missing a model_state mapping")
    arch = payload.get("temporal_model_arch")
    if arch is not None and arch != TEMPORAL_MODEL_ARCH:
        raise ValueError(f"Unsupported temporal checkpoint architecture: {arch!r}")
    if arch is None and "residual_scale_logit" not in state:
        raise ValueError("Expected the Hermite-conditioned residual ODE-RNN checkpoint")
    return TEMPORAL_MODEL_ARCH


def _temporal_checkpoint_dimensions(payload, arch):
    state = payload["model_state"]
    return {
        "core_dim": int(state["readout.bias"].shape[0]),
        "hidden_dim": int(state["forward_anchor_update.weight_hh"].shape[1]),
        "ode_hidden_dim": int(state["forward_ode_net.0.weight"].shape[0]),
    }


def _validate_temporal_checkpoint(
    payload, arch, *, core_dim, args, time_values, model_space, load_path
):
    dims = _temporal_checkpoint_dimensions(payload, arch)
    expected = {
        "core_dim": int(core_dim),
        "hidden_dim": int(args.temporal_hidden_dim),
        "ode_hidden_dim": int(args.temporal_ode_hidden_dim),
    }
    mismatches = [
        f"{name}: checkpoint={dims[name]}, config/runtime={expected[name]}"
        for name in expected
        if dims[name] != expected[name]
    ]
    saved_space = payload.get("temporal_model_space")
    saved_args = payload.get("args")
    if saved_space is None and isinstance(saved_args, dict):
        saved_space = saved_args.get("temporal_model_space")
    elif saved_space is None and saved_args is not None:
        saved_space = getattr(saved_args, "temporal_model_space", None)
    if saved_space is not None and str(saved_space).lower() != str(model_space).lower():
        mismatches.append(
            f"model_space: checkpoint={saved_space}, config={model_space}"
        )
    saved_times = payload.get("time_values")
    if saved_times is not None and (
        not _times_same_grid(
            saved_times, time_values, tol=float(args.query_time_match_tol)
        )
    ):
        if bool(getattr(args, "allow_checkpoint_data_manifest_mismatch", False)):
            print(
                "[checkpoint] allowing time grid mismatch for cross-protocol evaluation:",
                f"checkpoint_points={np.asarray(saved_times).size}",
                f"inference_points={np.asarray(time_values).size}",
            )
        else:
            mismatches.append("time_values differ from the current artifact/data grid")
    provenance_pairs = (
        ("ftm_manifest_path", getattr(args, "ftm_manifest_path", None)),
        ("data_manifest_path", getattr(args, "data_manifest_path", None)),
    )
    for name, current_path in provenance_pairs:
        saved_path = payload.get(name)
        if (
            saved_path
            and current_path
            and (
                os.path.realpath(str(saved_path)) != os.path.realpath(str(current_path))
            )
        ):
            if name == "data_manifest_path" and bool(
                getattr(args, "allow_checkpoint_data_manifest_mismatch", False)
            ):
                print(
                    "[checkpoint] allowing data manifest mismatch for cross-protocol evaluation:",
                    f"checkpoint={saved_path}",
                    f"inference={current_path}",
                )
                continue
            mismatches.append(f"{name}: checkpoint={saved_path}, config={current_path}")
    if mismatches:
        raise ValueError(
            f"Temporal checkpoint {load_path} is incompatible: " + "; ".join(mismatches)
        )
    return dims


def _temporal_checkpoint_payload(
    model, args, time_values, temporal_model_space, **extra
):
    residual_scale = None
    if hasattr(model, "residual_scale"):
        residual_scale = float(model.residual_scale().detach().cpu().item())
    payload = {
        "model_state": model.state_dict(),
        "temporal_model_arch": TEMPORAL_MODEL_ARCH,
        "temporal_model_space": str("core").lower(),
        "core_dim": int(model.core_dim),
        "hidden_dim": int(model.hidden_dim),
        "ode_hidden_dim": int(args.temporal_ode_hidden_dim),
        "residual_scale": residual_scale,
        "residual_scale_initial": getattr(model, "residual_scale_initial", None),
        "time_values": np.asarray(time_values, dtype=np.float32),
        "resolved_config": args.to_nested_dict(),
        "config_path": str(args.source_config_path),
        "ftm_manifest_path": str(args.ftm_manifest_path),
        "data_manifest_path": (
            None if not args.data_manifest_path else str(args.data_manifest_path)
        ),
    }
    payload.update(extra)
    return payload


def _infer_time_step(times_np):
    times = np.asarray(times_np, dtype=np.float64).reshape(-1)
    if int(times.size) <= 1:
        return 1.0
    deltas = np.diff(times)
    positive = deltas[deltas > 0.0]
    if positive.size <= 0:
        return 1.0
    return float(np.median(positive))


def _interpolate_tensor_time_linear(field_t, source_times_np, target_times_np):
    source_times = np.asarray(source_times_np, dtype=np.float64).reshape(-1)
    target_times = np.asarray(target_times_np, dtype=np.float64).reshape(-1)
    if _times_same_grid(source_times, target_times):
        return field_t
    if int(field_t.shape[1]) != int(source_times.size):
        raise ValueError(
            f"Time interpolation expects field_t.shape[1] == len(source_times), got {int(field_t.shape[1])} and {int(source_times.size)}"
        )
    if int(source_times.size) == 1:
        expand_shape = [int(field_t.shape[0]), int(target_times.size)] + list(
            field_t.shape[2:]
        )
        return field_t[:, :1].expand(*expand_shape)
    right = np.searchsorted(source_times, target_times, side="right")
    right = np.clip(right, 1, int(source_times.size) - 1)
    left = right - 1
    before = target_times <= float(source_times[0])
    after = target_times >= float(source_times[-1])
    left[before] = 0
    right[before] = 0
    left[after] = int(source_times.size) - 1
    right[after] = int(source_times.size) - 1
    denom = source_times[right] - source_times[left]
    weight_right = np.zeros_like(target_times, dtype=np.float32)
    valid = np.abs(denom) > 1e-12
    weight_right[valid] = (
        (target_times[valid] - source_times[left[valid]]) / denom[valid]
    ).astype(np.float32, copy=False)
    weight_right[before] = 0.0
    weight_right[after] = 0.0
    weight_left = 1.0 - weight_right
    left_idx_t = torch.as_tensor(left, dtype=torch.long, device=field_t.device)
    right_idx_t = torch.as_tensor(right, dtype=torch.long, device=field_t.device)
    left_vals = field_t.index_select(1, left_idx_t)
    right_vals = field_t.index_select(1, right_idx_t)
    shape = [1, int(target_times.size)] + [1] * (field_t.dim() - 2)
    w_left_t = torch.as_tensor(
        weight_left, dtype=field_t.dtype, device=field_t.device
    ).reshape(*shape)
    w_right_t = torch.as_tensor(
        weight_right, dtype=field_t.dtype, device=field_t.device
    ).reshape(*shape)
    return w_left_t * left_vals + w_right_t * right_vals


def _build_spatiotemporal_dense_query_fn_2d(
    field, source_times_np, target_times_np, coord_u, coord_v, *, periodic_u, periodic_v
):
    spatial_query_fn = build_dense_field_query_fn_2d(
        field, coord_u, coord_v, periodic_u=periodic_u, periodic_v=periodic_v
    )
    if _times_same_grid(source_times_np, target_times_np):
        return spatial_query_fn

    def query_fn(query_coords):
        sampled = spatial_query_fn(query_coords)
        return _interpolate_tensor_time_linear(
            sampled, source_times_np, target_times_np
        )

    return query_fn


def _parse_float_list_csv(text, name):
    values = []
    parts = (
        text
        if isinstance(text, (list, tuple, np.ndarray))
        else str(text).replace(";", ",").split(",")
    )
    for part in parts:
        if isinstance(part, str):
            part = part.strip()
            if not part:
                continue
        values.append(float(part))
    if not values:
        raise ValueError(f"{name} is empty.")
    for value in values:
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} values must be in (0, 1], got {value}.")
    return [float(v) for v in values]


def _parse_choice_list_csv(text, name, allowed, default=None):
    if isinstance(text, (list, tuple, np.ndarray)):
        raw_values = list(text)
    else:
        raw = str(text or "").strip()
        raw_values = [] if not raw else raw.replace(";", ",").split(",")
    if not raw_values:
        if default is None:
            raise ValueError(f"{name} is empty.")
        raw_values = (
            list(default)
            if isinstance(default, (list, tuple, np.ndarray))
            else str(default).replace(";", ",").split(",")
        )
    values = []
    for part in raw_values:
        part = str(part).strip().lower()
        if part:
            values.append(part)
    if not values:
        raise ValueError(f"{name} is empty.")
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(
            f"{name} has unsupported values {unknown}; allowed={sorted(allowed)}"
        )
    return values


def _temporal_time_mask_modes(args, kind="train"):
    allowed = {"identity", "random_drop_half", "blockmask"}
    if str(kind).lower() == "val":
        raw = getattr(args, "temporal_val_time_mask_mechanisms", "")
        if not raw:
            raw = getattr(args, "temporal_train_time_mask_mechanisms", "")
        return _parse_choice_list_csv(
            raw, "--temporal_val_time_mask_mechanisms", allowed
        )
    if str(kind).lower() == "train":
        raw = getattr(args, "temporal_train_time_mask_mechanisms", "")
        return _parse_choice_list_csv(
            raw, "--temporal_train_time_mask_mechanisms", allowed
        )
    raise ValueError(f"Unsupported kind={kind}")


def _parse_pde_snapshot_sample_indices(args):
    raw = getattr(args, "plot_pde_residual_sample_indices", "") or ""
    if isinstance(raw, (list, tuple, np.ndarray)):
        out = [int(value) for value in raw]
        if out:
            return out
    text = str(raw).strip()
    if text:
        out = []
        for token in text.replace(";", ",").split(","):
            token = token.strip()
            if token:
                out.append(int(token))
        if out:
            return out
    return [int(getattr(args, "plot_pde_residual_sample_index", 0))]


def _build_temporal_pde_kwargs(args, dataset_type):
    dataset_type = str(dataset_type).lower()
    if dataset_type == "ac":
        return dict(
            Lx=float(args.pde_Lx),
            Ly=float(args.pde_Ly),
            dt=0.01 if args.pde_dt is None else float(args.pde_dt),
            ac_lambda=float(args.pde_ac_lambda),
            epsilon=float(args.pde_ac_epsilon),
            fd_dx=None if float(args.pde_ac_fd_dx) <= 0.0 else float(args.pde_ac_fd_dx),
            fd_dy=None if float(args.pde_ac_fd_dy) <= 0.0 else float(args.pde_ac_fd_dy),
            fd_space_order=int(args.pde_ac_fd_space_order),
            fd_time_order=int(args.pde_ac_fd_time_order),
        )
    if dataset_type == "ns":
        return dict(
            Lx=float(args.pde_Lx),
            Ly=1.0 if float(args.pde_Ly) == 1.0 else float(args.pde_Ly),
            dt=0.1 if args.pde_dt is None else float(args.pde_dt),
            nu=float(args.pde_ns_nu),
            forcing_scale=float(args.pde_ns_forcing_scale),
            fd_dx=None if float(args.pde_ns_fd_dx) <= 0.0 else float(args.pde_ns_fd_dx),
            fd_dy=None if float(args.pde_ns_fd_dy) <= 0.0 else float(args.pde_ns_fd_dy),
            fd_space_order=int(args.pde_ns_fd_space_order),
            fd_time_order=int(args.pde_ns_fd_time_order),
        )
    if dataset_type == "heat3d":
        return dict(
            Lx=float(args.pde_Lx),
            Ly=float(args.pde_Ly),
            Lz=float(args.pde_Lz),
            dt=None if args.pde_dt is None else float(args.pde_dt),
            alpha=float(args.pde_heat3d_alpha),
            fd_dx=(
                None
                if float(args.pde_heat3d_fd_dx) <= 0.0
                else float(args.pde_heat3d_fd_dx)
            ),
            fd_dy=(
                None
                if float(args.pde_heat3d_fd_dy) <= 0.0
                else float(args.pde_heat3d_fd_dy)
            ),
            fd_dz=(
                None
                if float(args.pde_heat3d_fd_dz) <= 0.0
                else float(args.pde_heat3d_fd_dz)
            ),
            fd_space_order=int(args.pde_heat3d_fd_space_order),
            fd_time_order=int(getattr(args, "pde_heat3d_fd_time_order", 2) or 2),
        )
    if dataset_type == "ocean":
        return {}
    raise ValueError(
        f"Unsupported temporal PDE dataset_type={dataset_type!r}; expected AC, NS, heat3d, or ocean."
    )


def _temporal_pde_axis_periodicity(dataset_type):
    dataset_type = str(dataset_type).lower()
    if dataset_type in ("ac", "ns"):
        return (True, True)
    if dataset_type in ("heat3d", "ocean"):
        return (False, False)
    raise ValueError(
        f"Unsupported temporal PDE dataset_type={dataset_type!r}; expected AC, NS, heat3d, or ocean."
    )


def _can_use_ftm_continuous_pde_query(
    core_like, basis_function, shared_basis_multichannel=False
):
    if core_like is None or basis_function is None:
        return False
    if hasattr(basis_function, "mode_w"):
        return False
    if not hasattr(basis_function, "mode_u") or not hasattr(basis_function, "mode_v"):
        return False
    ndim = getattr(core_like, "ndim", None)
    if ndim is None:
        ndim = np.asarray(core_like).ndim
    ndim = int(ndim)
    if bool(shared_basis_multichannel):
        return ndim == 5
    return ndim == 4


def _build_ftm_continuous_query_fn_2d(
    *,
    core_t,
    basis_function,
    norm_stats,
    coord_uniques=None,
    shared_basis_multichannel=False,
):
    if not _can_use_ftm_continuous_pde_query(
        core_t, basis_function, shared_basis_multichannel=shared_basis_multichannel
    ):
        raise ValueError(
            "Direct FTM continuous PDE query requires a 2D core batch and basis_function.mode_u/mode_v."
        )
    core_t = torch.as_tensor(
        core_t, dtype=torch.float32, device=next(basis_function.parameters()).device
    )

    def query_fn(query_coords):
        with torch.no_grad():
            sampled_norm = _sample_field_with_mode_basis_2d(
                basis_function,
                core_t,
                query_coords,
                coord_uniques=coord_uniques,
                shared_basis_multichannel=bool(shared_basis_multichannel),
            )
            return inverse_normalize_tensor(sampled_norm, norm_stats)

    return query_fn


def _save_temporal_pde_snapshot(
    *,
    pred_np,
    data_t,
    norm_stats,
    pred_time_values_np,
    gt_time_values_np,
    target_time_values_np,
    coord_uniques,
    dataset_type,
    pde_kwargs,
    channel_order,
    point_sample_ratio,
    point_max_points,
    results_dir,
    step_tag,
    sample_indices,
    device,
    pred_core_np=None,
    basis_function=None,
    precomputed_basis=None,
    shared_basis_multichannel=False,
):
    if str(dataset_type).lower() in ("heat3d", "ocean"):
        return
    if int(len(sample_indices)) <= 0:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(results_dir, "pde_residual_snapshots")
    os.makedirs(out_dir, exist_ok=True)
    periodic_u, periodic_v = _temporal_pde_axis_periodicity(dataset_type)
    time_trim = int(pde_kwargs.get("fd_time_order", 2)) // 2
    dataset_type = str(dataset_type).lower()
    pred_time_values = np.asarray(pred_time_values_np, dtype=np.float32).reshape(-1)
    gt_time_values = np.asarray(gt_time_values_np, dtype=np.float32).reshape(-1)
    target_time_values = np.asarray(target_time_values_np, dtype=np.float32).reshape(-1)
    coord_u = torch.as_tensor(coord_uniques[0]).detach().float().cpu().reshape(-1)
    coord_v = torch.as_tensor(coord_uniques[1]).detach().float().cpu().reshape(-1)
    x_min = float(coord_u.min().item())
    x_max = float(coord_u.max().item())
    y_min = float(coord_v.min().item())
    y_max = float(coord_v.max().item())
    for sample_idx in sample_indices:
        if sample_idx < 0 or sample_idx >= int(pred_np.shape[0]):
            continue
        if dataset_type == "ns" and channel_order is None:
            continue
        pred_sample = torch.as_tensor(
            pred_np[sample_idx : sample_idx + 1], dtype=torch.float32, device=device
        )
        gt_sample = inverse_normalize_tensor(
            data_t[sample_idx : sample_idx + 1].to(device), norm_stats
        )
        if pred_sample.dim() == gt_sample.dim() - 1 and int(gt_sample.shape[-1]) == 1:
            pred_sample = pred_sample.unsqueeze(-1)
        elif (
            gt_sample.dim() == pred_sample.dim() - 1 and int(pred_sample.shape[-1]) == 1
        ):
            gt_sample = gt_sample.unsqueeze(-1)
        sample_coords = _build_plot_sample_coords(
            coord_uniques,
            sample_ratio=point_sample_ratio,
            max_points=point_max_points,
            device=device,
            seed_text=f"{results_dir}|{step_tag}|sample{sample_idx}",
        )
        if sample_coords is None:
            continue
        plot_coords = sample_coords.detach().float().cpu().numpy()
        plot_x = plot_coords[:, 0]
        plot_y = plot_coords[:, 1]
        plot_coord_uniques = coord_uniques
        plot_pde_kwargs = pde_kwargs
        use_direct_pred_query = _can_use_ftm_continuous_pde_query(
            pred_core_np,
            basis_function,
            shared_basis_multichannel=shared_basis_multichannel,
        )
        if use_direct_pred_query:
            pred_core_sample = torch.as_tensor(
                pred_core_np[sample_idx : sample_idx + 1],
                dtype=torch.float32,
                device=device,
            )
            pred_query_fn = _build_ftm_continuous_query_fn_2d(
                core_t=pred_core_sample,
                basis_function=basis_function,
                norm_stats=norm_stats,
                coord_uniques=coord_uniques,
                shared_basis_multichannel=shared_basis_multichannel,
            )
            if precomputed_basis is None:
                raise ValueError(
                    "Direct FTM PDE snapshot requires precomputed_basis for visualization decode."
                )
            pred_phys_full = _decode_core_tensor(
                pred_core_sample,
                precomputed_basis,
                norm_stats,
                shared_basis_multichannel=shared_basis_multichannel,
            )
        else:
            if not _times_same_grid(pred_time_values, target_time_values):
                pred_sample = _interpolate_tensor_time_linear(
                    pred_sample, pred_time_values, target_time_values
                )
            pred_phys_full = pred_sample
            pred_query_fn = build_dense_field_query_fn_2d(
                pred_phys_full,
                coord_uniques[0],
                coord_uniques[1],
                periodic_u=periodic_u,
                periodic_v=periodic_v,
            )
        if not _times_same_grid(gt_time_values, target_time_values):
            gt_sample = _interpolate_tensor_time_linear(
                gt_sample, gt_time_values, target_time_values
            )
        if int(pred_phys_full.shape[1]) < 3 or int(gt_sample.shape[1]) < 3:
            continue
        pred_terms = _compute_offgrid_point_residual_terms(
            query_fn=pred_query_fn,
            sample_coords=sample_coords,
            sample_t_coords=None,
            coord_uniques=plot_coord_uniques,
            pde_kwargs={
                **plot_pde_kwargs,
                "dataset_type": dataset_type,
                "channel_order": channel_order,
            },
        )
        if "residual_sq" not in pred_terms:
            continue
        pred_phys = pred_phys_full[0, time_trim:-time_trim].detach().float().cpu()
        gt_phys = gt_sample[0, time_trim:-time_trim].detach().float().cpu()
        if pred_phys.dim() == 3:
            pred_phys = pred_phys.unsqueeze(-1)
            gt_phys = gt_phys.unsqueeze(-1)
        display_channel = 0
        if channel_order is not None:
            order = tuple((str(x) for x in channel_order))
            if "omega" in order:
                display_channel = order.index("omega")
            elif "h" in order:
                display_channel = order.index("h")
            elif "b" in order:
                display_channel = order.index("b")
        pred_loss = pred_terms["residual_sq"][0].detach().float().cpu()
        if int(pred_loss.shape[0]) <= 0:
            continue
        time_ids = np.linspace(
            0,
            int(pred_loss.shape[0]) - 1,
            num=min(3, int(pred_loss.shape[0])),
            dtype=int,
        )
        sample_dir = os.path.join(out_dir, step_tag, f"sample_{sample_idx:03d}")
        latest_dir = os.path.join(out_dir, "latest", f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)
        for tid in time_ids.tolist():
            center_tid = tid + time_trim
            gt_img = gt_phys[tid, ..., display_channel].numpy()
            pred_img = pred_phys[tid, ..., display_channel].numpy()
            err_img = pred_img - gt_img
            pred_loss_img = pred_loss[tid].numpy()
            field_abs = max(
                float(np.abs(pred_img).max()), float(np.abs(gt_img).max()), 1e-12
            )
            err_abs = max(float(np.abs(err_img).max()), 1e-12)
            loss_abs = max(float(pred_loss_img.max()), 1e-12)
            fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), squeeze=False)
            panels = [
                (gt_img, "GT", "RdBu_r", -field_abs, field_abs),
                (pred_img, "Pred", "RdBu_r", -field_abs, field_abs),
                (err_img, "Pred-GT", "RdBu_r", -err_abs, err_abs),
            ]
            for col, (img, title, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[0, col]
                im = ax.imshow(
                    img,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    extent=(y_min, y_max, x_min, x_max),
                )
                ax.set_title(title)
                ax.set_aspect("equal")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax = axes[1, 0]
            im = ax.scatter(
                plot_y,
                plot_x,
                c=pred_loss_img,
                s=12,
                cmap="viridis",
                vmin=0.0,
                vmax=loss_abs,
            )
            ax.set_title("Pred PDE residual squared")
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(x_min, x_max)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax = axes[1, 1]
            log_img = np.log10(np.clip(pred_loss_img, a_min=1e-12, a_max=None))
            im = ax.scatter(plot_y, plot_x, c=log_img, s=12, cmap="magma")
            ax.set_title("log10 residual squared")
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(x_min, x_max)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax = axes[1, 2]
            ax.axis("off")
            ax.text(
                0.02,
                0.98,
                "\n".join(
                    [
                        f"dataset={dataset_type}",
                        f"step={step_tag}",
                        f"sample={sample_idx}",
                        f"time_idx={center_tid}",
                        f"mean_res2={float(np.mean(pred_loss_img)):.3e}",
                        f"max_res2={float(np.max(pred_loss_img)):.3e}",
                    ]
                ),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="0.6",
                ),
            )
            for ax in axes.reshape(-1):
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(
                f"PDE Residual Snapshot | {dataset_type} | {step_tag} | sample={sample_idx} | time={center_tid}",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            filename = f"{dataset_type}_{step_tag}_sample{sample_idx:03d}_time{center_tid:04d}.png"
            fig.savefig(
                os.path.join(sample_dir, filename), dpi=200, bbox_inches="tight"
            )
            fig.savefig(
                os.path.join(
                    latest_dir,
                    f"{dataset_type}_latest_sample{sample_idx:03d}_time{center_tid:04d}.png",
                ),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


def _infer_dataset_type(
    data_manifest_path=None, data_manifest=None, ftm_manifest=None, num_channels=None
):
    text_parts = []
    if data_manifest_path:
        text_parts.append(str(data_manifest_path))
    if isinstance(data_manifest, dict):
        text_parts.extend(
            [
                str(data_manifest.get("data_name", "")),
                str(data_manifest.get("source_dir", "")),
            ]
        )
    if isinstance(ftm_manifest, dict):
        text_parts.extend(
            [
                str(ftm_manifest.get("data_name", "")),
                str(ftm_manifest.get("source_data_manifest_path", "")),
            ]
        )
    merged = " | ".join(text_parts).upper()
    merged_norm = merged.replace("-", "_")
    for part in text_parts:
        token = os.path.basename(str(part).strip()).upper().replace("-", "_")
        if (
            token.startswith("NS")
            or "NS2D" in token
            or "UVOMEGA" in token
            or ("OMEGAONLY" in token)
        ):
            return "NS"
    if "NS2D" in merged_norm or "UVOMEGA" in merged_norm or "OMEGAONLY" in merged_norm:
        return "NS"
    if "AC" in merged_norm:
        return "AC"
    if "HEAT3D" in merged_norm or "HEAT_3D" in merged_norm:
        return "heat3d"
    if "OCEAN" in merged_norm:
        return "ocean"
    return None


def _resolve_temporal_pde_dataset_type(
    requested_type,
    *,
    data_manifest_path=None,
    data_manifest=None,
    ftm_manifest=None,
    num_channels=None,
):
    requested = str(requested_type or "auto").strip().lower()
    if requested == "none":
        return None
    if requested in ("ac", "ns", "heat3d", "ocean"):
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported pde_dataset_type={requested_type!r}")
    inferred = _infer_dataset_type(
        data_manifest_path=data_manifest_path,
        data_manifest=data_manifest,
        ftm_manifest=ftm_manifest,
        num_channels=num_channels,
    )
    if inferred is None:
        raise ValueError(
            f"Cannot infer pde_dataset_type from manifest_path={data_manifest_path!r}, data_name={(None if not isinstance(data_manifest, dict) else data_manifest.get('data_name'))!r}, ftm_data_name={(None if ftm_manifest is None else ftm_manifest.get('data_name'))!r}. Set pde.dataset_type explicitly to ac, ns, heat3d, ocean, or none."
        )
    return str(inferred).lower()


def _times_same_grid(times_a, times_b, tol=1e-07):
    a = np.asarray(times_a, dtype=np.float64).reshape(-1)
    b = np.asarray(times_b, dtype=np.float64).reshape(-1)
    return a.shape == b.shape and bool(np.allclose(a, b, atol=float(tol), rtol=0.0))


def _load_physical_time_coords(metadata_path, fallback_times):
    """Read physical time coordinates when a dataset stores normalized time indices."""
    try:
        payload = np.load(metadata_path, allow_pickle=True).item()
        data_meta = payload.get("data", payload) if isinstance(payload, dict) else {}
        values = (
            data_meta.get("t_coords_physical") if isinstance(data_meta, dict) else None
        )
        if values is not None:
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if values.size == np.asarray(fallback_times).size:
                return values
    except Exception:
        pass
    return np.asarray(fallback_times, dtype=np.float32).reshape(-1)


def _artifact_time_values(ftm_manifest, fallback_times):
    """Use the evaluated dataset's time grid for a per-time spatial artifact."""
    fallback = np.asarray(fallback_times, dtype=np.float32).reshape(-1)
    if fallback.size:
        return fallback
    return np.asarray(
        ftm_manifest.get("coordinates", {}).get("t", []), dtype=np.float32
    ).reshape(-1)


def _time_observation_mask(mask_np):
    mask_arr = np.asarray(mask_np, dtype=np.float32)
    if mask_arr.ndim == 4:
        mask_arr = mask_arr[..., None]
    return np.any(mask_arr > 0, axis=tuple(range(2, mask_arr.ndim)))


def _build_inference_time_mask(num_times, template, seed):
    num_times = int(num_times)
    keep = np.ones(num_times, dtype=bool)
    template = str(template).lower()
    if template == "all":
        return keep
    if template == "random_drop_half":
        rng = np.random.default_rng(int(seed))
        keep[:] = rng.random(num_times) >= 0.5
        if num_times > 0:
            keep[0] = True
            keep[-1] = True
        if not np.any(keep):
            keep[0] = True
        return keep
    raise ValueError(f"Unsupported inference_time_template={template}")


def _sample_block_time_keep(num_times, keep_ratio, rng):
    num_times = int(num_times)
    keep = np.ones(num_times, dtype=bool)
    if num_times <= 0:
        return keep
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    if keep_ratio >= 1.0 or num_times <= 2:
        keep[0] = True
        keep[-1] = True
        return keep
    drop_count = int(round((1.0 - keep_ratio) * num_times))
    drop_count = max(1, min(drop_count, max(num_times - 2, 1)))
    max_start = max(num_times - 1 - drop_count, 1)
    start = int(rng.integers(1, max_start + 1))
    end = min(start + drop_count, num_times - 1)
    keep[start:end] = False
    keep[0] = True
    keep[-1] = True
    if not np.any(keep):
        keep[0] = True
    return keep


def _sample_temporal_keep_mask(num_times, mode, keep_ratio, rng):
    num_times = int(num_times)
    keep_ratio = float(keep_ratio)
    mode = str(mode).lower()
    if mode == "identity":
        keep = np.ones(num_times, dtype=bool)
    elif mode == "random_drop_half":
        keep = rng.random(num_times) >= 0.5
    elif mode == "blockmask":
        keep = _sample_block_time_keep(num_times, keep_ratio=keep_ratio, rng=rng)
    else:
        raise ValueError(f"Unsupported temporal mask mode={mode}")
    if num_times > 0:
        keep[0] = True
        keep[-1] = True
    if num_times > 0 and (not np.any(keep)):
        keep[0] = True
    return keep


def _apply_time_keep_to_mask(mask_np, time_keep):
    inference_mask = np.asarray(mask_np, dtype=np.float32).copy()
    time_keep = np.asarray(time_keep, dtype=bool).reshape(-1)
    if inference_mask.shape[1] != time_keep.size:
        raise ValueError(
            f"mask time length {inference_mask.shape[1]} != time_keep length {time_keep.size}"
        )
    inference_mask[:, ~time_keep, ...] = 0.0
    return inference_mask


def _ensure_mask_5d(mask_np):
    mask = np.asarray(mask_np, dtype=np.float32)
    if mask.ndim == 4:
        mask = mask[..., None]
    if mask.ndim not in (5, 6):
        raise ValueError(
            f"Expected mask shape (N,T,H,W[,C]) or (N,T,D,H,W[,C]), got {mask.shape}"
        )
    return mask


def _ensure_mask_for_data(mask_np, spatial_dim):
    mask = np.asarray(mask_np, dtype=np.float32)
    spatial_dim = int(spatial_dim)
    if spatial_dim == 2:
        return _ensure_mask_5d(mask)
    if mask.ndim == 5:
        mask = mask[..., None]
    if mask.ndim != 6:
        raise ValueError(f"Expected 3D mask shape (N,T,D,H,W[,C]), got {mask.shape}")
    return mask


def _block_mask_to_observation_ratio(mask_np, target_ratio, rng):
    mask = _ensure_mask_5d(mask_np).copy()
    for batch_idx in range(int(mask.shape[0])):
        keep = _sample_block_time_keep(
            int(mask.shape[1]), keep_ratio=float(target_ratio), rng=rng
        )
        mask[batch_idx, ~keep, ...] = 0.0
    return mask


def _apply_time_mask_mode(mask_np, target_ratio, rng, mode, args):
    mode = str(mode).lower()
    mask = _ensure_mask_5d(mask_np)
    if mode == "identity":
        return mask.copy()
    if mode in ("random_drop_half",):
        out = mask.copy()
        for batch_idx in range(int(out.shape[0])):
            keep = _sample_temporal_keep_mask(
                int(out.shape[1]), mode=mode, keep_ratio=float(target_ratio), rng=rng
            )
            out[batch_idx, ~keep, ...] = 0.0
        return out
    if mode == "blockmask":
        return _block_mask_to_observation_ratio(mask, target_ratio, rng)
    raise ValueError(f"Unsupported time mask mode={mode}")


def _apply_time_mask_mixture(mask_np, target_ratio, rng, modes, args):
    modes = [str(x).lower() for x in modes]
    mask = _ensure_mask_5d(mask_np)
    if len(modes) == 1:
        return (
            _apply_time_mask_mode(mask, target_ratio, rng, modes[0], args),
            [modes[0]] * int(mask.shape[0]),
        )
    out = np.zeros_like(mask, dtype=np.float32)
    chosen = []
    for batch_idx in range(int(mask.shape[0])):
        mode = str(modes[int(rng.integers(len(modes)))])
        chosen.append(mode)
        out[batch_idx : batch_idx + 1] = _apply_time_mask_mode(
            mask[batch_idx : batch_idx + 1], target_ratio, rng, mode, args
        )
    return (out, chosen)


def _mode_count_text(modes):
    if not modes:
        return "none"
    counts = {}
    for mode in modes:
        counts[str(mode)] = counts.get(str(mode), 0) + 1
    return ",".join((f"{key}:{counts[key]}" for key in sorted(counts)))


def _resolve_inference_obs_ratio(mask_np, requested_ratio):
    requested_ratio = float(requested_ratio)
    if requested_ratio > 0.0:
        return requested_ratio
    return float(np.mean(_time_observation_mask(mask_np)))


def _flatten_core_batch(core_t):
    return core_t.reshape(int(core_t.shape[0]), int(core_t.shape[1]), -1)


def _unflatten_core_batch(flat_t, core_shape):
    return flat_t.reshape(int(flat_t.shape[0]), int(flat_t.shape[1]), *core_shape)


def _build_cubic_hermite_baseline_and_gate(
    anchor_core, source_times, obs_time_mask, target_times, match_tol=1e-07
):
    device = anchor_core.device
    dtype = anchor_core.dtype
    B = int(anchor_core.shape[0])
    target_times = target_times.reshape(-1).to(device=device, dtype=dtype)
    source_times = source_times.reshape(-1).to(device=device, dtype=dtype)
    flat = _flatten_core_batch(anchor_core)
    flat_dim = int(flat.shape[-1])
    Q = int(target_times.numel())
    baseline = torch.zeros((B, Q, flat_dim), dtype=dtype, device=device)
    gate = torch.zeros((B, Q), dtype=dtype, device=device)
    pi_t = torch.as_tensor(np.pi, dtype=dtype, device=device)
    for batch_idx in range(B):
        obs_idx = torch.nonzero(obs_time_mask[batch_idx], as_tuple=False).reshape(-1)
        if obs_idx.numel() == 0:
            continue
        obs_times = source_times.index_select(0, obs_idx)
        obs_flat = flat[batch_idx].index_select(0, obs_idx)
        if obs_idx.numel() == 1:
            baseline[batch_idx] = obs_flat[0:1].expand(Q, flat_dim)
            continue
        slopes = torch.zeros_like(obs_flat)
        dt0 = torch.clamp(obs_times[1] - obs_times[0], min=1e-08)
        dt1 = torch.clamp(obs_times[-1] - obs_times[-2], min=1e-08)
        slopes[0] = (obs_flat[1] - obs_flat[0]) / dt0
        slopes[-1] = (obs_flat[-1] - obs_flat[-2]) / dt1
        if int(obs_idx.numel()) > 2:
            dt_mid = torch.clamp(obs_times[2:] - obs_times[:-2], min=1e-08).unsqueeze(
                -1
            )
            slopes[1:-1] = (obs_flat[2:] - obs_flat[:-2]) / dt_mid
        right = torch.searchsorted(obs_times, target_times, right=False)
        right = torch.clamp(right, 0, int(obs_times.numel()) - 1)
        left = torch.clamp(right - 1, 0, int(obs_times.numel()) - 1)
        left_t = obs_times.index_select(0, left)
        right_t = obs_times.index_select(0, right)
        left_v = obs_flat.index_select(0, left)
        right_v = obs_flat.index_select(0, right)
        left_s = slopes.index_select(0, left)
        right_s = slopes.index_select(0, right)
        exact_left = torch.abs(target_times - left_t) <= float(match_tol)
        exact_right = torch.abs(target_times - right_t) <= float(match_tol)
        before_first = target_times <= obs_times[0] + float(match_tol)
        after_last = target_times >= obs_times[-1] - float(match_tol)
        inside = ~before_first & ~after_last & (right > left)
        baseline_cur = torch.empty((Q, flat_dim), dtype=dtype, device=device)
        baseline_cur[before_first] = obs_flat[0]
        baseline_cur[after_last] = obs_flat[-1]
        baseline_cur[exact_left] = left_v[exact_left]
        baseline_cur[exact_right] = right_v[exact_right]
        if inside.any():
            dt = torch.clamp(right_t[inside] - left_t[inside], min=1e-08)
            tau = (target_times[inside] - left_t[inside]) / dt
            tau2 = tau * tau
            tau3 = tau2 * tau
            h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0
            h10 = tau3 - 2.0 * tau2 + tau
            h01 = -2.0 * tau3 + 3.0 * tau2
            h11 = tau3 - tau2
            baseline_cur[inside] = (
                h00.unsqueeze(-1) * left_v[inside]
                + h10.unsqueeze(-1) * dt.unsqueeze(-1) * left_s[inside]
                + h01.unsqueeze(-1) * right_v[inside]
                + h11.unsqueeze(-1) * dt.unsqueeze(-1) * right_s[inside]
            )
            gate[batch_idx, inside] = torch.sin(pi_t * tau) ** 2
        unresolved = ~(before_first | after_last | exact_left | exact_right | inside)
        if unresolved.any():
            baseline_cur[unresolved] = left_v[unresolved]
        baseline[batch_idx] = baseline_cur
    return (_unflatten_core_batch(baseline, tuple(anchor_core.shape[2:])), gate)


def _run_temporal_head_on_grid(
    *,
    temporal_method,
    temporal_model,
    anchor_series_np,
    source_times_np,
    obs_time_mask_np,
    target_times_np,
    args,
    device,
):
    return _run_temporal_residual_model(
        model=temporal_model,
        anchor_series_np=anchor_series_np,
        source_times_np=source_times_np,
        obs_time_mask_np=obs_time_mask_np,
        target_times_np=target_times_np,
        args=args,
        device=device,
    )


class CoreODERNN(nn.Module):
    """Bidirectional GRU updates and latent ODE steps used by the residual head."""

    def __init__(self, core_dim, hidden_dim, ode_hidden_dim):
        super().__init__()
        self.core_dim = int(core_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = self.core_dim + 2
        self.forward_anchor_update = nn.GRUCell(self.input_dim, self.hidden_dim)
        self.backward_anchor_update = nn.GRUCell(self.input_dim, self.hidden_dim)
        self.forward_ode_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, int(ode_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(ode_hidden_dim), self.hidden_dim),
            nn.Tanh(),
        )
        self.backward_ode_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, int(ode_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(ode_hidden_dim), self.hidden_dim),
            nn.Tanh(),
        )
        self.readout = nn.Linear(2 * self.hidden_dim, self.core_dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _ode_step(self, h, current_time, dt, ode_net):
        if float(torch.abs(dt).item()) <= 0.0:
            return h
        t_feat = torch.full(
            (int(h.shape[0]), 1), float(current_time), dtype=h.dtype, device=h.device
        )
        dh = ode_net(torch.cat([h, t_feat], dim=-1))
        return h + dt * dh

    def _anchor_features(self, obs_flat, obs_times, obs_ptr):
        zero = torch.zeros((1, 1), dtype=obs_flat.dtype, device=obs_flat.device)
        if int(obs_ptr) <= 0:
            delta_prev = zero
        else:
            delta_prev = (obs_times[obs_ptr] - obs_times[obs_ptr - 1]).reshape(1, 1)
        if int(obs_ptr) + 1 >= int(obs_times.numel()):
            delta_next = zero
        else:
            delta_next = (obs_times[obs_ptr + 1] - obs_times[obs_ptr]).reshape(1, 1)
        return torch.cat(
            [obs_flat[obs_ptr : obs_ptr + 1], delta_prev, delta_next], dim=-1
        )

    def _run_direction(self, obs_flat, obs_times, target_times, anchor_update, ode_net):
        num_obs = int(obs_times.numel())
        num_targets = int(target_times.numel())
        hidden_trace = torch.zeros(
            (num_targets, self.hidden_dim), dtype=obs_flat.dtype, device=obs_flat.device
        )
        if num_obs <= 1:
            return hidden_trace
        h = torch.zeros(
            (1, self.hidden_dim), dtype=obs_flat.dtype, device=obs_flat.device
        )
        h = anchor_update(self._anchor_features(obs_flat, obs_times, obs_ptr=0), h)
        current_time = obs_times[0]
        for obs_ptr in range(num_obs - 1):
            next_time = obs_times[obs_ptr + 1]
            inside = torch.nonzero(
                (target_times >= current_time) & (target_times <= next_time),
                as_tuple=False,
            ).reshape(-1)
            for qi in inside.tolist():
                qt = target_times[qi]
                if float((qt - current_time).abs().item()) > 0.0:
                    h = self._ode_step(
                        h,
                        current_time=current_time,
                        dt=qt - current_time,
                        ode_net=ode_net,
                    )
                    current_time = qt
                hidden_trace[qi] = h[0]
            if float((next_time - current_time).abs().item()) > 0.0:
                h = self._ode_step(
                    h,
                    current_time=current_time,
                    dt=next_time - current_time,
                    ode_net=ode_net,
                )
                current_time = next_time
            h = anchor_update(
                self._anchor_features(obs_flat, obs_times, obs_ptr=obs_ptr + 1), h
            )
        return hidden_trace


class TemporalResidualODERNN(CoreODERNN):
    """Hermite-conditioned bidirectional ODE-RNN residual corrector.

    The cubic-Hermite estimate is the identity path.  The learned branch sees
    that estimate explicitly and predicts only a bounded residual correction.
    A zero-initialized final readout makes the initial model exactly equal to
    cubic Hermite even though the residual scale is initialized to a small,
    trainable positive value.
    """

    def __init__(self, core_dim, hidden_dim, ode_hidden_dim):
        super().__init__(
            core_dim=core_dim, hidden_dim=hidden_dim, ode_hidden_dim=ode_hidden_dim
        )
        self.residual_fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + self.core_dim, self.hidden_dim), nn.SiLU()
        )
        self.readout = nn.Linear(self.hidden_dim, self.core_dim)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        initial_scale = 0.05
        initial_logit = float(np.log(initial_scale / (1.0 - initial_scale)))
        self.residual_scale_logit = nn.Parameter(
            torch.tensor(initial_logit, dtype=torch.float32)
        )
        self.residual_scale_initial = float(initial_scale)

    def residual_scale(self):
        return torch.sigmoid(self.residual_scale_logit)

    def forward(
        self, anchor_core, source_times, obs_time_mask, target_times, baseline_core=None
    ):
        flat = _flatten_core_batch(anchor_core)
        B, _, D = flat.shape
        Q = int(target_times.numel())
        source_times = source_times.reshape(-1).to(
            device=anchor_core.device, dtype=anchor_core.dtype
        )
        target_times = target_times.reshape(-1).to(
            device=anchor_core.device, dtype=anchor_core.dtype
        )
        if baseline_core is None:
            baseline_core, _ = _build_cubic_hermite_baseline_and_gate(
                anchor_core=anchor_core,
                source_times=source_times,
                obs_time_mask=obs_time_mask,
                target_times=target_times,
            )
        baseline_flat = _flatten_core_batch(baseline_core).to(
            device=anchor_core.device, dtype=anchor_core.dtype
        )
        if tuple(baseline_flat.shape) != (B, Q, D):
            raise ValueError(
                f"Hermite baseline shape does not match the temporal residual query: baseline={tuple(baseline_flat.shape)} expected={(B, Q, D)}"
            )
        out = torch.zeros((B, Q, D), dtype=anchor_core.dtype, device=anchor_core.device)
        for batch_idx in range(B):
            obs_idx = torch.nonzero(obs_time_mask[batch_idx], as_tuple=False).reshape(
                -1
            )
            if obs_idx.numel() == 0:
                continue
            obs_times = source_times.index_select(0, obs_idx)
            obs_flat = flat[batch_idx].index_select(0, obs_idx)
            forward_hidden = self._run_direction(
                obs_flat,
                obs_times,
                target_times,
                self.forward_anchor_update,
                self.forward_ode_net,
            )
            reverse_hidden = torch.zeros_like(forward_hidden)
            if int(obs_idx.numel()) > 1:
                reverse_hidden_rev = self._run_direction(
                    torch.flip(obs_flat, dims=[0]),
                    -torch.flip(obs_times, dims=[0]),
                    -torch.flip(target_times, dims=[0]),
                    self.backward_anchor_update,
                    self.backward_ode_net,
                )
                reverse_hidden = torch.flip(reverse_hidden_rev, dims=[0])
            fused = self.residual_fusion(
                torch.cat(
                    [forward_hidden, reverse_hidden, baseline_flat[batch_idx]], dim=-1
                )
            )
            out[batch_idx] = self.readout(fused)
        out = self.residual_scale().to(dtype=out.dtype) * out
        return _unflatten_core_batch(out, tuple(anchor_core.shape[2:]))


def _temporal_model_class_for_checkpoint_arch(checkpoint_arch):
    if checkpoint_arch != TEMPORAL_MODEL_ARCH:
        raise ValueError(f"Unsupported temporal architecture: {checkpoint_arch!r}")
    return TemporalResidualODERNN


def _masked_time_mse(pred, target, time_mask):
    if time_mask.dtype != torch.bool:
        time_mask = time_mask > 0
    if pred.dim() == 4:
        weights = time_mask.unsqueeze(-1).unsqueeze(-1)
    elif pred.dim() == 5:
        weights = time_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    elif pred.dim() == 6:
        weights = time_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported pred dim for masked mse: {pred.dim()}")
    weights = weights.to(dtype=pred.dtype, device=pred.device)
    diff2 = (pred - target) ** 2
    denom = torch.clamp(weights.sum() * float(np.prod(diff2.shape[2:])), min=1.0)
    return torch.sum(diff2 * weights) / denom


def _decode_core_tensor(
    core_t, precomputed_basis, norm_stats, shared_basis_multichannel=False
):
    pred_norm = _decode_with_precomputed_basis(
        precomputed_basis,
        core_t,
        shared_basis_multichannel=bool(shared_basis_multichannel),
    )
    return inverse_normalize_tensor(pred_norm, norm_stats)


def _series_feature_dim(series_shape):
    return int(np.prod(series_shape[2:]))


def _temporal_residual_correction(gate, residual_core):
    """Return the actual gated correction that is added to cubic Hermite."""
    gate_shape = [int(gate.shape[0]), int(gate.shape[1])] + [1] * (
        residual_core.dim() - 2
    )
    return gate.reshape(*gate_shape) * residual_core


def _apply_temporal_residual_to_baseline(baseline_core, gate, residual_core):
    if residual_core.dim() != baseline_core.dim():
        raise ValueError(
            f"Residual rank {residual_core.dim()} does not match baseline rank {baseline_core.dim()}."
        )
    return baseline_core + _temporal_residual_correction(gate, residual_core)


def _solve_anchor_cores_for_dataset(
    *,
    data_t,
    mask_np,
    core_train_np,
    mean_core_np,
    train_data_norm,
    train_mask_np,
    precomputed_basis,
    basis_function,
    ind_input,
    rank,
    shared_basis_multichannel,
    args,
    device,
    batch_size,
    log_prefix=None,
):
    anchor_core_batches = []
    anchor_prior_batches = []
    num_samples = int(data_t.shape[0])
    num_batches = (num_samples + int(batch_size) - 1) // int(batch_size)
    mask_t_all = torch.as_tensor(mask_np, dtype=torch.float32)
    for batch_no, start in enumerate(range(0, num_samples, int(batch_size)), start=1):
        end = min(start + int(batch_size), num_samples)
        batch_idx = torch.arange(start, end, dtype=torch.long)
        data_batch = data_t[batch_idx].to(device)
        mask_batch = get_mask_batch(
            mask_t_all, batch_idx, data_batch=data_batch, device=device
        )
        with torch.no_grad():
            prior_core = _mean_prior_core_batch(
                mean_core_np, int(data_batch.shape[0]), device=device
            ).to(dtype=data_batch.dtype)
            prior_pred_norm = _decode_with_precomputed_basis(
                precomputed_basis,
                prior_core,
                shared_basis_multichannel=shared_basis_multichannel,
            )
            residual_target = data_batch - prior_pred_norm
            if hasattr(basis_function, "mode_w") and shared_basis_multichannel:
                residual_core = (
                    solve_shared_basis_multichannel_core_3d_from_sparse_mask(
                        basis=basis_function,
                        y=residual_target,
                        mask=mask_batch,
                        coord_uniques=ind_input,
                        rank=rank,
                        lambda_ridge=float(args.lambda_core_ridge),
                        beta_tv=float(args.beta_core_tv),
                        max_cg_iter=int(args.cg_max_iter),
                        cg_tol=float(args.cg_tol),
                        initial_core=None,
                        precomputed_basis=precomputed_basis,
                    )
                )
            elif hasattr(basis_function, "mode_w"):
                residual_core = ridge_tv_solve_core_3d_from_sparse_mask(
                    basis=basis_function,
                    y=residual_target,
                    mask=mask_batch,
                    coord_uniques=ind_input,
                    rank=rank,
                    lambda_ridge=float(args.lambda_core_ridge),
                    beta_tv=float(args.beta_core_tv),
                    max_cg_iter=int(args.cg_max_iter),
                    cg_tol=float(args.cg_tol),
                    initial_core=None,
                    precomputed_basis=precomputed_basis,
                )
            elif shared_basis_multichannel:
                residual_core = (
                    solve_shared_basis_multichannel_core_2d_from_sparse_mask(
                        basis=basis_function,
                        y=residual_target,
                        mask=mask_batch,
                        coord_uniques=ind_input,
                        rank=rank,
                        lambda_ridge=float(args.lambda_core_ridge),
                        beta_tv=float(args.beta_core_tv),
                        max_cg_iter=int(args.cg_max_iter),
                        cg_tol=float(args.cg_tol),
                        initial_core=None,
                        precomputed_basis=precomputed_basis,
                    )
                )
            else:
                residual_core = ridge_tv_solve_core_2d_from_sparse_mask(
                    basis=basis_function,
                    y=residual_target,
                    mask=mask_batch,
                    coord_uniques=ind_input,
                    rank=rank,
                    lambda_ridge=float(args.lambda_core_ridge),
                    beta_tv=float(args.beta_core_tv),
                    max_cg_iter=int(args.cg_max_iter),
                    cg_tol=float(args.cg_tol),
                    initial_core=None,
                    precomputed_basis=precomputed_basis,
                )
            anchor_core = prior_core + residual_core
        anchor_core_batches.append(
            anchor_core.detach().cpu().numpy().astype(np.float32)
        )
        anchor_prior_batches.append(
            prior_core.detach().cpu().numpy().astype(np.float32)
        )
        if log_prefix:
            print(
                f"{log_prefix} batch {batch_no:03d}/{num_batches:03d} | samples {start:03d}:{end:03d}"
            )
    return (
        np.concatenate(anchor_core_batches, axis=0),
        np.concatenate(anchor_prior_batches, axis=0),
    )


def _nanmean_or_nan(values):
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _temporal_val_cache_key(time_mask_mode, obs_ratio):
    return f"{str(time_mask_mode).lower()}|{float(obs_ratio):.6f}"


def _temporal_val_cache_paths(cache_dir, time_mask_mode, obs_ratio):
    key = _temporal_val_cache_key(time_mask_mode, obs_ratio)
    safe = key.replace("|", "_p").replace(".", "p")
    return {
        "anchor_core_path": os.path.join(cache_dir, f"{safe}_anchor_core.npy"),
        "obs_time_mask_path": os.path.join(cache_dir, f"{safe}_obs_time_mask.npy"),
        "meta_path": os.path.join(cache_dir, f"{safe}_meta.json"),
    }


def _evaluate_temporal_residual_model_case(
    *,
    model,
    data_eval_t,
    mask_eval_np,
    time_values,
    core_train_np,
    mean_core_np,
    train_data_norm,
    train_mask_np,
    precomputed_basis,
    basis_function,
    ind_input,
    rank,
    norm_stats,
    shared_basis_multichannel,
    args,
    device,
    batch_size,
    temporal_model_space,
    obs_ratio,
    time_mask_mode,
    coord_uniques=None,
    dataset_type="ac",
    pde_kwargs=None,
    pde_channel_order_names=None,
    results_dir=None,
    step_tag=None,
    cached_anchor_case=None,
):
    model.eval()
    time_values = np.asarray(time_values, dtype=np.float32)
    if cached_anchor_case is not None:
        obs_time_mask_np = np.asarray(
            cached_anchor_case["obs_time_mask_np"], dtype=bool
        )
        anchor_core_path = cached_anchor_case.get("anchor_core_path")
        if anchor_core_path:
            anchor_core_np = np.load(anchor_core_path, mmap_mode="r")
        else:
            anchor_core_np = np.asarray(
                cached_anchor_case["anchor_core_np"], dtype=np.float32
            )
        mask_ratio_actual = float(cached_anchor_case["mask_ratio_actual"])
    else:
        obs_time_mask_np = _time_observation_mask(mask_eval_np)
        anchor_core_np, _ = _solve_anchor_cores_for_dataset(
            data_t=data_eval_t,
            mask_np=mask_eval_np,
            core_train_np=core_train_np,
            mean_core_np=mean_core_np,
            train_data_norm=train_data_norm,
            train_mask_np=train_mask_np,
            precomputed_basis=precomputed_basis,
            basis_function=basis_function,
            ind_input=ind_input,
            rank=rank,
            shared_basis_multichannel=shared_basis_multichannel,
            args=args,
            device=device,
            batch_size=batch_size,
            log_prefix=None,
        )
        mask_ratio_actual = float(np.mean(np.asarray(mask_eval_np, dtype=np.float32)))
    kept_per_sample = np.sum(obs_time_mask_np, axis=1)
    missing_per_sample = np.sum(~obs_time_mask_np, axis=1)
    print(
        f"[temporal val case] mode={str(time_mask_mode).lower()} p={float(obs_ratio):.3f} kept_mean={float(np.mean(kept_per_sample)):.2f}/{int(data_eval_t.shape[1])}"
    )
    anchor_series_np = anchor_core_np
    _, _, _, final_series_np, _ = _run_temporal_residual_model(
        model=model,
        anchor_series_np=anchor_series_np,
        source_times_np=time_values,
        obs_time_mask_np=obs_time_mask_np,
        target_times_np=time_values,
        args=args,
        device=device,
    )
    pred_np = _decode_core_only(
        core_np=final_series_np,
        precomputed_basis=precomputed_basis,
        norm_stats=norm_stats,
        batch_size=batch_size,
        save_dtype=np.float32,
        device=device,
        num_channels=int(data_eval_t.shape[-1]),
        shared_basis_multichannel=shared_basis_multichannel,
    )
    full_time_mask = np.ones(int(data_eval_t.shape[1]), dtype=bool)
    rmse_full = _compute_time_subset_rmse_batched(
        pred_np=pred_np,
        data_t=data_eval_t,
        norm_stats=norm_stats,
        time_mask=full_time_mask,
        batch_size=batch_size,
        device=device,
    )
    rel_l2_full = _compute_time_subset_rel_l2_batched(
        pred_np=pred_np,
        data_t=data_eval_t,
        norm_stats=norm_stats,
        time_mask=full_time_mask,
        batch_size=batch_size,
        device=device,
    )
    rmse_missing = _compute_time_subset_rmse_batched(
        pred_np=pred_np,
        data_t=data_eval_t,
        norm_stats=norm_stats,
        time_mask=~obs_time_mask_np,
        batch_size=batch_size,
        device=device,
    )
    rel_l2_missing = _compute_time_subset_rel_l2_batched(
        pred_np=pred_np,
        data_t=data_eval_t,
        norm_stats=norm_stats,
        time_mask=~obs_time_mask_np,
        batch_size=batch_size,
        device=device,
    )
    pde_metrics = {}
    if (
        pde_kwargs
        and str(dataset_type).lower() in ("ac", "ns", "heat3d", "ocean")
        and (coord_uniques is not None)
    ):
        pde_metrics = _compute_temporal_pde_batched(
            pred_np=pred_np,
            data_t=data_eval_t,
            norm_stats=norm_stats,
            time_mask=full_time_mask,
            pred_time_values_np=time_values,
            gt_time_values_np=time_values,
            target_time_values_np=time_values,
            batch_size=batch_size,
            device=device,
            coord_uniques=coord_uniques,
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
            channel_order=pde_channel_order_names,
            point_sample_ratio=float(args.pde_point_sample_ratio),
            point_max_points=int(args.pde_point_max_points),
            pred_core_np=final_series_np,
            basis_function=basis_function,
            shared_basis_multichannel=shared_basis_multichannel,
        )
        if (
            results_dir is not None
            and step_tag is not None
            and bool(args.plot_pde_residual_on_log)
        ):
            _save_temporal_pde_snapshot(
                pred_np=pred_np,
                data_t=data_eval_t,
                norm_stats=norm_stats,
                pred_time_values_np=time_values,
                gt_time_values_np=time_values,
                target_time_values_np=time_values,
                coord_uniques=coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=pde_channel_order_names,
                point_sample_ratio=float(args.pde_point_sample_ratio),
                point_max_points=int(args.pde_point_max_points),
                results_dir=results_dir,
                step_tag=step_tag,
                sample_indices=_parse_pde_snapshot_sample_indices(args),
                device=device,
                pred_core_np=final_series_np,
                basis_function=basis_function,
                precomputed_basis=precomputed_basis,
                shared_basis_multichannel=shared_basis_multichannel,
            )
    return {
        "obs_ratio": float(obs_ratio),
        "obs_ratio_actual": float(np.mean(obs_time_mask_np)),
        "mask_ratio_actual": mask_ratio_actual,
        "time_mask_mode": str(time_mask_mode).lower(),
        "rmse_full": float(rmse_full),
        "rel_l2_full": float(rel_l2_full),
        "rmse_missing": float(rmse_missing),
        "rel_l2_missing": float(rel_l2_missing),
        "kept_frames_mean": float(np.mean(kept_per_sample)),
        "missing_frames_mean": float(np.mean(missing_per_sample)),
        **pde_metrics,
    }


def _evaluate_temporal_residual_model(
    *,
    model,
    data_eval_t,
    mask_eval_np,
    time_values,
    core_train_np,
    mean_core_np,
    train_data_norm,
    train_mask_np,
    precomputed_basis,
    basis_function,
    ind_input,
    rank,
    norm_stats,
    shared_basis_multichannel,
    args,
    device,
    batch_size,
    temporal_model_space,
    obs_ratio_list,
    coord_uniques=None,
    dataset_type="ac",
    pde_kwargs=None,
    pde_channel_order_names=None,
    results_dir=None,
    step_tag=None,
    val_anchor_cache=None,
    val_anchor_cache_dir=None,
):
    val_modes = _temporal_time_mask_modes(args, kind="val")
    val_rng = np.random.default_rng(int(args.temporal_val_seed))
    metric_list = []
    for obs_ratio in obs_ratio_list:
        for mode in val_modes:
            cache_key = _temporal_val_cache_key(mode, obs_ratio)
            cached_anchor_case = (
                None if val_anchor_cache is None else val_anchor_cache.get(cache_key)
            )
            case_mask_np = None
            if cached_anchor_case is None:
                case_mask_np = _apply_time_mask_mode(
                    mask_eval_np, float(obs_ratio), val_rng, str(mode), args
                )
                obs_time_mask_np = _time_observation_mask(case_mask_np)
                mask_ratio_actual = float(
                    np.mean(np.asarray(case_mask_np, dtype=np.float32))
                )
                anchor_core_np, _ = _solve_anchor_cores_for_dataset(
                    data_t=data_eval_t,
                    mask_np=case_mask_np,
                    core_train_np=core_train_np,
                    mean_core_np=mean_core_np,
                    train_data_norm=train_data_norm,
                    train_mask_np=train_mask_np,
                    precomputed_basis=precomputed_basis,
                    basis_function=basis_function,
                    ind_input=ind_input,
                    rank=rank,
                    shared_basis_multichannel=shared_basis_multichannel,
                    args=args,
                    device=device,
                    batch_size=batch_size,
                    log_prefix=None,
                )
                if val_anchor_cache_dir is not None:
                    paths = _temporal_val_cache_paths(
                        val_anchor_cache_dir, mode, obs_ratio
                    )
                    np.save(
                        paths["anchor_core_path"],
                        np.asarray(anchor_core_np, dtype=np.float32),
                    )
                    np.save(
                        paths["obs_time_mask_path"],
                        np.asarray(obs_time_mask_np, dtype=np.uint8),
                    )
                    with open(paths["meta_path"], "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "time_mask_mode": str(mode).lower(),
                                "obs_ratio": float(obs_ratio),
                                "mask_ratio_actual": float(mask_ratio_actual),
                            },
                            f,
                            indent=2,
                        )
                    anchor_core_path = paths["anchor_core_path"]
                    obs_time_mask_path = paths["obs_time_mask_path"]
                else:
                    anchor_core_path = None
                    obs_time_mask_path = None
                cached_anchor_case = {
                    "time_mask_mode": str(mode).lower(),
                    "obs_ratio": float(obs_ratio),
                    "mask_ratio_actual": float(mask_ratio_actual),
                    "obs_time_mask_np": np.asarray(obs_time_mask_np, dtype=bool),
                    "anchor_core_path": anchor_core_path,
                    "anchor_core_np": (
                        None
                        if anchor_core_path is not None
                        else np.asarray(anchor_core_np, dtype=np.float32)
                    ),
                    "obs_time_mask_path": obs_time_mask_path,
                }
                if val_anchor_cache is not None:
                    val_anchor_cache[cache_key] = cached_anchor_case
                print(
                    f"[temporal val cache] store mode={str(mode).lower()} p={float(obs_ratio):.3f}"
                    + (
                        f" path={anchor_core_path}"
                        if anchor_core_path is not None
                        else " storage=memory"
                    )
                )
            else:
                print(
                    f"[temporal val cache] hit mode={str(mode).lower()} p={float(obs_ratio):.3f}"
                )
            case_step_tag = None
            if results_dir is not None and step_tag is not None:
                case_step_tag = (
                    f"{step_tag}_{str(mode).lower()}_p{float(obs_ratio):.3f}".replace(
                        ".", "p"
                    )
                )
            metric_list.append(
                _evaluate_temporal_residual_model_case(
                    model=model,
                    data_eval_t=data_eval_t,
                    mask_eval_np=mask_eval_np if case_mask_np is None else case_mask_np,
                    time_values=time_values,
                    core_train_np=core_train_np,
                    mean_core_np=mean_core_np,
                    train_data_norm=train_data_norm,
                    train_mask_np=train_mask_np,
                    precomputed_basis=precomputed_basis,
                    basis_function=basis_function,
                    ind_input=ind_input,
                    rank=rank,
                    norm_stats=norm_stats,
                    shared_basis_multichannel=shared_basis_multichannel,
                    args=args,
                    device=device,
                    batch_size=batch_size,
                    temporal_model_space="core",
                    obs_ratio=float(obs_ratio),
                    time_mask_mode=str(mode).lower(),
                    coord_uniques=coord_uniques,
                    dataset_type=dataset_type,
                    pde_kwargs=pde_kwargs,
                    pde_channel_order_names=pde_channel_order_names,
                    results_dir=results_dir,
                    step_tag=case_step_tag,
                    cached_anchor_case=cached_anchor_case,
                )
            )
    rmse_full = _nanmean_or_nan((item["rmse_full"] for item in metric_list))
    rel_l2_full = float(
        np.mean([item["rel_l2_full"] for item in metric_list], dtype=np.float64)
    )
    rmse_missing = _nanmean_or_nan((item["rmse_missing"] for item in metric_list))
    rel_l2_missing = _nanmean_or_nan((item["rel_l2_missing"] for item in metric_list))
    kept_frames_mean = _nanmean_or_nan(
        (item["kept_frames_mean"] for item in metric_list)
    )
    missing_frames_mean = _nanmean_or_nan(
        (item["missing_frames_mean"] for item in metric_list)
    )
    aggregate = {}
    for item in metric_list:
        for key, value in item.items():
            if key in {
                "obs_ratio",
                "obs_ratio_actual",
                "mask_ratio_actual",
                "time_mask_mode",
                "rmse_full",
                "rel_l2_full",
                "rmse_missing",
                "rel_l2_missing",
                "kept_frames_mean",
                "missing_frames_mean",
            }:
                continue
            if not isinstance(value, (int, float, np.integer, np.floating)):
                continue
            aggregate.setdefault(key, []).append(value)
    aggregate = {key: _nanmean_or_nan(values) for key, values in aggregate.items()}
    return {
        "rmse_full": rmse_full,
        "rel_l2_full": rel_l2_full,
        "rmse_missing": rmse_missing,
        "rel_l2_missing": rel_l2_missing,
        "num_samples": int(data_eval_t.shape[0]),
        "kept_frames_mean": kept_frames_mean,
        "missing_frames_mean": missing_frames_mean,
        "per_obs_ratio": metric_list,
        **aggregate,
    }


def _train_temporal_residual_model(
    *,
    model,
    core_train_np,
    train_data_t,
    train_mask_np,
    obs_ratio_list,
    time_values,
    val_data_t,
    val_mask_np,
    mean_core_np,
    train_data_norm,
    precomputed_basis,
    basis_function,
    ind_input,
    rank,
    norm_stats,
    shared_basis_multichannel,
    args,
    device,
    coord_uniques,
    dataset_type,
    pde_kwargs,
    pde_channel_order_names,
    experiment_root_dir,
    best_model_path=None,
    temporal_model_space="core",
):
    model.train()
    core_fit_np = core_train_np
    print(
        "[temporal train] start:",
        f"samples={int(core_train_np.shape[0])}",
        f"fit={int(core_fit_np.shape[0])}",
        f"val={(0 if val_data_t is None else int(val_data_t.shape[0]))}",
        f"arch={TEMPORAL_MODEL_ARCH}",
        "supervision=missing-frame core residual MSE",
        f"train_time_modes={','.join(_temporal_time_mask_modes(args, kind='train'))}",
        f"val_time_modes={','.join(_temporal_time_mask_modes(args, kind='val'))}",
        f"epochs={int(args.temporal_train_epochs)}",
        f"batch_size={max(int(args.temporal_train_batch_size), 1)}",
        f"lr={float(args.temporal_train_lr):.3e}",
    )
    scale_params = (
        [model.residual_scale_logit] if hasattr(model, "residual_scale_logit") else []
    )
    scale_param_ids = {id(param) for param in scale_params}
    trainable_params = [
        param for param in model.parameters() if id(param) not in scale_param_ids
    ]
    optimizer_param_groups = [{"params": trainable_params}]
    if scale_params:
        optimizer_param_groups.append({"params": scale_params, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=float(args.temporal_train_lr),
        weight_decay=float(args.temporal_train_weight_decay),
    )
    rng = np.random.default_rng(int(args.temporal_train_seed))
    train_batch_size = max(int(args.temporal_train_batch_size), 1)
    eval_batch_size = max(int(args.temporal_train_batch_size), 1)
    num_samples = int(core_fit_np.shape[0])
    target_times_t = torch.as_tensor(time_values, dtype=torch.float32, device=device)
    train_time_modes = _temporal_time_mask_modes(args, kind="train")
    if not any((str(mode).lower() != "identity" for mode in train_time_modes)):
        raise ValueError(
            "Temporal residual training requires at least one missing-time mask mechanism; an identity-only protocol gives the residual branch no trainable target."
        )
    selection_metric = "rel_l2_missing"
    best_selection_score = float("inf")
    best_epoch = -1
    best_state = None
    configured_core_loss_weight = float(args.temporal_core_loss_weight)
    if configured_core_loss_weight < 0.0:
        raise ValueError("temporal core_loss_weight must be nonnegative")
    direct_residual_loss_weight = configured_core_loss_weight
    if direct_residual_loss_weight == 0.0:
        raise ValueError("Temporal training requires at least one positive loss weight")
    print(
        f"[temporal train] direct core residual supervision weight={direct_residual_loss_weight:g}"
    )
    val_anchor_cache = {}
    val_anchor_cache_dir = None
    if val_data_t is not None:
        val_anchor_cache_dir = os.path.join(experiment_root_dir, "val_anchor_cache")
        os.makedirs(val_anchor_cache_dir, exist_ok=True)
        print(f"[temporal val] anchor cache dir: {val_anchor_cache_dir}")

    def _evaluate_and_select(epoch_number, step_tag, label):
        nonlocal best_selection_score, best_epoch, best_state
        metrics = _evaluate_temporal_residual_model(
            model=model,
            data_eval_t=val_data_t,
            mask_eval_np=val_mask_np,
            time_values=time_values,
            core_train_np=core_train_np,
            mean_core_np=mean_core_np,
            train_data_norm=train_data_norm,
            train_mask_np=train_mask_np,
            precomputed_basis=precomputed_basis,
            basis_function=basis_function,
            ind_input=ind_input,
            rank=rank,
            norm_stats=norm_stats,
            shared_basis_multichannel=shared_basis_multichannel,
            args=args,
            device=device,
            batch_size=eval_batch_size,
            temporal_model_space="core",
            obs_ratio_list=obs_ratio_list,
            coord_uniques=coord_uniques,
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
            pde_channel_order_names=pde_channel_order_names,
            results_dir=experiment_root_dir,
            step_tag=step_tag,
            val_anchor_cache=val_anchor_cache,
            val_anchor_cache_dir=val_anchor_cache_dir,
        )
        rmse_full = float(metrics["rmse_full"])
        rel_l2_full = float(metrics["rel_l2_full"])
        rmse_missing = float(metrics["rmse_missing"])
        rel_l2_missing = float(metrics["rel_l2_missing"])
        print(
            f"[temporal val] {label} {int(epoch_number):03d}/{int(args.temporal_train_epochs):03d} rmse_full={rmse_full:.6e} relL2_full={rel_l2_full:.6e} rmse_missing={rmse_missing:.6e} relL2_missing={rel_l2_missing:.6e} kept_mean={float(metrics['kept_frames_mean']):.2f}/{int(core_train_np.shape[1])}"
        )
        for ratio_metric in metrics["per_obs_ratio"]:
            print(
                f"[temporal val detail] {label} {int(epoch_number):03d} mode={str(ratio_metric.get('time_mask_mode', 'metadata'))} p={float(ratio_metric['obs_ratio']):.3f} p_actual={float(ratio_metric.get('obs_ratio_actual', float('nan'))):.3f} rmse_full={float(ratio_metric['rmse_full']):.6e} relL2_full={float(ratio_metric['rel_l2_full']):.6e} rmse_missing={float(ratio_metric['rmse_missing']):.6e} relL2_missing={float(ratio_metric['rel_l2_missing']):.6e}"
            )
        selection_score = float(metrics[selection_metric])
        if not np.isfinite(selection_score):
            if best_state is None:
                raise ValueError(
                    "Epoch-0 Hermite validation rel_l2_missing is not finite. Configure at least one validation time-mask mechanism with missing frames; model selection never falls back to a full-time metric."
                )
            print(
                f"[temporal val] {label} {int(epoch_number):03d} produced non-finite {selection_metric}; keeping epoch {best_epoch:03d}."
            )
            return metrics
        if best_state is None or selection_score < best_selection_score:
            best_selection_score = selection_score
            best_epoch = int(epoch_number)
            best_state = copy.deepcopy(model.state_dict())
            if best_model_path:
                best_payload = _temporal_checkpoint_payload(
                    model,
                    args,
                    time_values,
                    "core",
                    best_rel_l2_full=rel_l2_full,
                    best_rel_l2_missing=rel_l2_missing,
                    best_rmse_missing=rmse_missing,
                    selection_metric=selection_metric,
                    selection_score=selection_score,
                    best_epoch=best_epoch,
                    best_is_hermite_baseline=bool(best_epoch == 0),
                    obs_ratio_list=np.asarray(obs_ratio_list, dtype=np.float32),
                    val_mask_mode=list(_temporal_time_mask_modes(args, kind="val")),
                    val_metrics=metrics,
                )
                best_payload["model_state"] = best_state
                torch.save(best_payload, best_model_path)
            print(
                f"[temporal val] new best: epoch={best_epoch:03d} {selection_metric}={best_selection_score:.6e}"
                + (f" path={best_model_path}" if best_model_path else "")
            )
        return metrics

    if val_data_t is not None:
        print(
            "[temporal val] evaluating epoch 000 zero-residual cubic Hermite baseline"
        )
        _evaluate_and_select(
            epoch_number=0, step_tag="epoch000_hermite_baseline", label="baseline"
        )
    for epoch in range(int(args.temporal_train_epochs)):
        model.train()
        perm = rng.permutation(num_samples)
        epoch_loss = 0.0
        epoch_residual = 0.0
        batch_count = 0
        num_batches = (num_samples + train_batch_size - 1) // train_batch_size
        for start in range(0, num_samples, train_batch_size):
            end = min(start + train_batch_size, num_samples)
            batch_idx = perm[start:end]
            batch_idx_t = torch.as_tensor(batch_idx, dtype=torch.long)
            target_core = torch.as_tensor(
                core_fit_np[batch_idx], dtype=torch.float32, device=device
            )
            obs_ratio = float(obs_ratio_list[int(rng.integers(len(obs_ratio_list)))])
            data_batch = train_data_t.index_select(0, batch_idx_t)
            original_mask_batch_np = np.asarray(
                train_mask_np[batch_idx], dtype=np.float32
            )
            mask_batch_np, chosen_modes = _apply_time_mask_mixture(
                original_mask_batch_np, obs_ratio, rng, train_time_modes, args
            )
            anchor_core_np, _ = _solve_anchor_cores_for_dataset(
                data_t=data_batch,
                mask_np=mask_batch_np,
                core_train_np=core_train_np,
                mean_core_np=mean_core_np,
                train_data_norm=train_data_norm,
                train_mask_np=train_mask_np,
                precomputed_basis=precomputed_basis,
                basis_function=basis_function,
                ind_input=ind_input,
                rank=rank,
                shared_basis_multichannel=shared_basis_multichannel,
                args=args,
                device=device,
                batch_size=int(data_batch.shape[0]),
                log_prefix=None,
            )
            anchor_core = torch.as_tensor(
                anchor_core_np, dtype=torch.float32, device=device
            )
            obs_mask = torch.as_tensor(
                _time_observation_mask(mask_batch_np), dtype=torch.bool, device=device
            )
            anchor_series = anchor_core
            baseline_core, gate = _build_cubic_hermite_baseline_and_gate(
                anchor_core=anchor_series,
                source_times=target_times_t,
                obs_time_mask=obs_mask,
                target_times=target_times_t,
                match_tol=float(args.query_time_match_tol),
            )
            residual_core = model(
                anchor_core=anchor_series,
                source_times=target_times_t,
                obs_time_mask=obs_mask,
                target_times=target_times_t,
                baseline_core=baseline_core,
            )
            missing_mask = ~obs_mask
            loss = torch.zeros((), dtype=torch.float32, device=device)
            residual_loss = None
            if direct_residual_loss_weight > 0.0:
                pred_residual = _temporal_residual_correction(gate, residual_core)
                target_residual = (target_core - baseline_core).detach()
                residual_loss = _masked_time_mse(
                    pred_residual, target_residual, missing_mask
                )
                loss = loss + direct_residual_loss_weight * residual_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            if residual_loss is not None:
                epoch_residual += float(residual_loss.item())
            batch_count += 1
            if batch_count == 1 or batch_count % 10 == 0 or batch_count == num_batches:
                residual_log = ""
                if residual_loss is not None:
                    residual_log = f" residual={float(residual_loss.item()):.6e}"
                print(
                    f"[temporal train] epoch {epoch + 1:03d}/{int(args.temporal_train_epochs):03d} batch {batch_count:03d}/{num_batches:03d} loss={float(loss.item()):.6e} p={obs_ratio:.3f} mask_modes={_mode_count_text(chosen_modes)}{residual_log}"
                )
        train_loss = epoch_loss / max(batch_count, 1)
        residual_epoch_log = f" residual={epoch_residual / max(batch_count, 1):.6e}"
        print(
            f"[temporal train] epoch {epoch + 1:03d}/{int(args.temporal_train_epochs):03d} loss={train_loss:.6e}{residual_epoch_log}"
        )
        if val_data_t is not None:
            _evaluate_and_select(
                epoch_number=epoch + 1, step_tag=f"epoch{epoch + 1:03d}", label="epoch"
            )
    if best_state is not None:
        model.load_state_dict(best_state)
        selected_kind = (
            "cubic Hermite baseline" if best_epoch == 0 else "trained residual"
        )
        print(
            f"[temporal train] loaded best checkpoint from epoch {best_epoch:03d} ({selected_kind}) with {selection_metric}={best_selection_score:.6e}"
        )
        if best_epoch == 0:
            print(
                f"[temporal train] no trained epoch improved validation {selection_metric}; using cubic Hermite."
            )
    model.eval()
    return model


def _run_temporal_residual_model(
    *,
    model,
    anchor_series_np,
    source_times_np,
    obs_time_mask_np,
    target_times_np,
    args,
    device,
):
    model.eval()
    anchor_core_t = torch.as_tensor(
        anchor_series_np, dtype=torch.float32, device=device
    )
    source_times_t = torch.as_tensor(
        source_times_np, dtype=torch.float32, device=device
    )
    obs_mask_t = torch.as_tensor(obs_time_mask_np, dtype=torch.bool, device=device)
    target_times_t = torch.as_tensor(
        target_times_np, dtype=torch.float32, device=device
    )
    baseline_core_t, gate_t = _build_cubic_hermite_baseline_and_gate(
        anchor_core=anchor_core_t,
        source_times=source_times_t,
        obs_time_mask=obs_mask_t,
        target_times=target_times_t,
        match_tol=float(args.query_time_match_tol),
    )
    with torch.no_grad():
        residual_core_t = model(
            anchor_core=anchor_core_t,
            source_times=source_times_t,
            obs_time_mask=obs_mask_t,
            target_times=target_times_t,
            baseline_core=baseline_core_t,
        )
    core_final_t = _apply_temporal_residual_to_baseline(
        baseline_core_t, gate_t, residual_core_t
    )
    matched_anchor_count = 0
    if int(args.override_anchor_times) == 1:
        target = np.asarray(target_times_np, dtype=np.float64).reshape(-1)
        source = np.asarray(source_times_np, dtype=np.float64).reshape(-1)
        for source_idx, source_time in enumerate(source):
            matches = np.flatnonzero(
                np.abs(target - source_time) <= float(args.query_time_match_tol)
            )
            if matches.size == 0:
                continue
            target_idx = int(matches[0])
            matched_anchor_count += 1
            for batch_idx in range(int(core_final_t.shape[0])):
                if bool(obs_time_mask_np[batch_idx, source_idx]):
                    core_final_t[batch_idx, target_idx] = anchor_core_t[
                        batch_idx, source_idx
                    ]
    return (
        baseline_core_t.detach().cpu().numpy().astype(np.float32),
        gate_t.detach().cpu().numpy().astype(np.float32),
        residual_core_t.detach().cpu().numpy().astype(np.float32),
        core_final_t.detach().cpu().numpy().astype(np.float32),
        matched_anchor_count,
    )


def _decode_core_batches(
    core_np,
    precomputed_basis,
    norm_stats,
    batch_size,
    save_dtype,
    device,
    channel_names,
    data,
    mask_np,
    results_dir,
    time_indices,
    n_plot_samples,
    shared_basis_multichannel=False,
    spatial_dim=2,
):
    pred_batches = []
    metric_accum = None
    num_samples = int(core_np.shape[0])
    num_channels = int(data.shape[-1])
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        core_batch = torch.as_tensor(
            core_np[start:end], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            pred_norm = _decode_with_precomputed_basis(
                precomputed_basis,
                core_batch,
                shared_basis_multichannel=bool(shared_basis_multichannel),
            )
            pred_phys = inverse_normalize_tensor(pred_norm, norm_stats)
            gt_phys = inverse_normalize_tensor(data[start:end].to(device), norm_stats)
        diff = pred_phys - gt_phys
        mask_batch = torch.as_tensor(
            mask_np[start:end], dtype=torch.float32, device=device
        )
        if mask_batch.shape[-1] == 1 and num_channels > 1:
            mask_batch = mask_batch.expand(*mask_batch.shape[:-1], num_channels)
        metric_accum = _accumulate_channel_metrics(
            metric_accum, diff, gt_phys, mask_batch
        )
        pred_np = pred_phys.detach().cpu().numpy()
        if num_channels == 1:
            pred_np = pred_np.squeeze(-1)
        pred_batches.append(pred_np.astype(save_dtype))
    pred_all = np.concatenate(pred_batches, axis=0)
    plot_count = min(max(int(n_plot_samples), 0), num_samples)
    for sample_idx in range(plot_count):
        gt_phys_np = (
            inverse_normalize_tensor(
                data[sample_idx : sample_idx + 1].to(device), norm_stats
            )
            .cpu()
            .numpy()[0]
        )
        if num_channels == 1:
            gt_phys_np = gt_phys_np.squeeze(-1)
        mask_sample = np.asarray(mask_np[sample_idx], dtype=np.float32)
        if num_channels == 1 and mask_sample.shape[-1] == 1:
            mask_sample = mask_sample[..., 0]
        _plot_sample(
            sample_idx=sample_idx,
            pred=pred_all[sample_idx],
            gt=gt_phys_np,
            mask=mask_sample,
            time_indices=time_indices,
            out_path=os.path.join(
                results_dir, "plots", "fields", f"sample_{sample_idx:03d}.png"
            ),
            channel_names=channel_names,
            spatial_dim=spatial_dim,
        )
    return (pred_all, metric_accum)


def _decode_core_only(
    core_np,
    precomputed_basis,
    norm_stats,
    batch_size,
    save_dtype,
    device,
    num_channels,
    shared_basis_multichannel=False,
):
    pred_batches = []
    num_samples = int(core_np.shape[0])
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        core_batch = torch.as_tensor(
            core_np[start:end], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            pred_norm = _decode_with_precomputed_basis(
                precomputed_basis,
                core_batch,
                shared_basis_multichannel=bool(shared_basis_multichannel),
            )
            pred_phys = inverse_normalize_tensor(pred_norm, norm_stats)
        pred_np = pred_phys.detach().cpu().numpy()
        if int(num_channels) == 1:
            pred_np = pred_np.squeeze(-1)
        pred_batches.append(pred_np.astype(save_dtype))
    return np.concatenate(pred_batches, axis=0)


def _compute_time_subset_rel_l2_batched(
    pred_np, data_t, norm_stats, time_mask, batch_size, device
):
    time_mask = np.asarray(time_mask, dtype=bool)
    if not np.any(time_mask):
        return float("nan")
    total_num = 0.0
    total_den = 0.0
    num_samples = int(pred_np.shape[0])
    for start in range(0, num_samples, int(batch_size)):
        end = min(start + int(batch_size), num_samples)
        pred_t = torch.as_tensor(pred_np[start:end], dtype=torch.float32, device=device)
        gt_t = inverse_normalize_tensor(data_t[start:end].to(device), norm_stats)
        if pred_t.dim() == gt_t.dim() - 1 and int(gt_t.shape[-1]) == 1:
            pred_t = pred_t.unsqueeze(-1)
        elif gt_t.dim() == pred_t.dim() - 1 and int(pred_t.shape[-1]) == 1:
            gt_t = gt_t.unsqueeze(-1)
        if pred_t.shape != gt_t.shape:
            raise ValueError(
                f"Shape mismatch for batched relL2 subset: pred={tuple(pred_t.shape)} gt={tuple(gt_t.shape)}"
            )
        if time_mask.ndim == 1:
            mask_batch = time_mask.reshape(-1)
            diff = pred_t[:, mask_batch] - gt_t[:, mask_batch]
            total_num += float(torch.sum(diff**2).item())
            total_den += float(torch.sum(gt_t[:, mask_batch] ** 2).item())
        elif time_mask.ndim == 2:
            mask_batch = torch.as_tensor(
                time_mask[start:end], dtype=torch.bool, device=device
            )
            if tuple(mask_batch.shape) != (int(pred_t.shape[0]), int(pred_t.shape[1])):
                raise ValueError(
                    f"Per-sample time_mask shape {tuple(mask_batch.shape)} does not match batch shape {(int(pred_t.shape[0]), int(pred_t.shape[1]))}"
                )
            mask_expand = mask_batch.view(
                int(pred_t.shape[0]), int(pred_t.shape[1]), *[1] * (pred_t.dim() - 2)
            )
            total_num += float(torch.sum((pred_t - gt_t) ** 2 * mask_expand).item())
            total_den += float(torch.sum(gt_t**2 * mask_expand).item())
        else:
            raise ValueError(f"time_mask must be 1D or 2D, got shape={time_mask.shape}")
    return float(np.sqrt(total_num / max(total_den, 1e-12)))


def _compute_time_subset_rmse_batched(
    pred_np, data_t, norm_stats, time_mask, batch_size, device
):
    time_mask = np.asarray(time_mask, dtype=bool)
    if not np.any(time_mask):
        return float("nan")
    total_sse = 0.0
    total_count = 0.0
    num_samples = int(pred_np.shape[0])
    for start in range(0, num_samples, int(batch_size)):
        end = min(start + int(batch_size), num_samples)
        pred_t = torch.as_tensor(pred_np[start:end], dtype=torch.float32, device=device)
        gt_t = inverse_normalize_tensor(data_t[start:end].to(device), norm_stats)
        if pred_t.dim() == gt_t.dim() - 1 and int(gt_t.shape[-1]) == 1:
            pred_t = pred_t.unsqueeze(-1)
        elif gt_t.dim() == pred_t.dim() - 1 and int(pred_t.shape[-1]) == 1:
            gt_t = gt_t.unsqueeze(-1)
        if pred_t.shape != gt_t.shape:
            raise ValueError(
                f"Shape mismatch for batched RMSE subset: pred={tuple(pred_t.shape)} gt={tuple(gt_t.shape)}"
            )
        if time_mask.ndim == 1:
            mask_batch = time_mask.reshape(-1)
            diff = pred_t[:, mask_batch] - gt_t[:, mask_batch]
            total_sse += float(torch.sum(diff**2).item())
            total_count += float(diff.numel())
        elif time_mask.ndim == 2:
            mask_batch = torch.as_tensor(
                time_mask[start:end], dtype=torch.bool, device=device
            )
            if tuple(mask_batch.shape) != (int(pred_t.shape[0]), int(pred_t.shape[1])):
                raise ValueError(
                    f"Per-sample time_mask shape {tuple(mask_batch.shape)} does not match batch shape {(int(pred_t.shape[0]), int(pred_t.shape[1]))}"
                )
            mask_expand = mask_batch.view(
                int(pred_t.shape[0]), int(pred_t.shape[1]), *[1] * (pred_t.dim() - 2)
            )
            total_sse += float(torch.sum((pred_t - gt_t) ** 2 * mask_expand).item())
            total_count += float(mask_batch.sum().item() * np.prod(pred_t.shape[2:]))
        else:
            raise ValueError(f"time_mask must be 1D or 2D, got shape={time_mask.shape}")
    return float(np.sqrt(total_sse / max(total_count, 1.0)))


def _compute_temporal_protocol_metrics_batched(
    *, predictions, data_t, norm_stats, mask_np, batch_size, device
):
    """Compute support/query and sensor/off-sensor RMSE for field predictions.

    ``predictions`` maps a metric suffix to a physical-space prediction array.
    The empty suffix produces the standard protocol keys; for example,
    ``{"_cubic": pred}`` produces ``offsensor_rmse_on_query_time_cubic``.
    """
    accumulators = {
        str(suffix): protocol_rmse_accumulator("temporal_protocol")
        for suffix in predictions
    }
    num_samples = int(data_t.shape[0])
    spatial_dim = int(data_t.dim()) - 3
    mask = _ensure_mask_for_data(mask_np, spatial_dim=spatial_dim)
    for start in range(0, num_samples, max(int(batch_size), 1)):
        end = min(start + max(int(batch_size), 1), num_samples)
        gt_batch = (
            inverse_normalize_tensor(data_t[start:end].to(device), norm_stats)
            .detach()
            .cpu()
            .numpy()
        )
        mask_batch = np.asarray(mask[start:end], dtype=np.float32)
        for suffix, pred_all in predictions.items():
            pred_batch = np.asarray(pred_all[start:end], dtype=np.float32)
            if pred_batch.ndim == gt_batch.ndim - 1 and int(gt_batch.shape[-1]) == 1:
                pred_batch = pred_batch[..., None]
            if tuple(pred_batch.shape) != tuple(gt_batch.shape):
                raise ValueError(
                    f"Protocol metric prediction shape {pred_batch.shape} does not match GT {gt_batch.shape}."
                )
            for local_idx in range(end - start):
                accumulate_protocol_rmse(
                    accumulators[str(suffix)],
                    protocol="temporal_protocol",
                    pred=pred_batch[local_idx],
                    gt=gt_batch[local_idx],
                    sample_mask=mask_batch[local_idx],
                )
    metrics = {}
    for suffix, accumulator in accumulators.items():
        metrics.update(
            {f"{key}{suffix}": value for key, value in accumulator.metrics().items()}
        )
    return metrics


def _resolve_evaluation_ground_truth_path(
    *, requested_path, data_manifest_path, data_manifest, observation_data_path
):
    """Resolve an optional clean test path while preserving ordinary manifests."""
    split_info = (data_manifest or {}).get("splits", {}).get("test", {})
    manifest_clean_path = (
        split_info.get("clean_data_path")
        or split_info.get("query_ground_truth_data_path")
        or (data_manifest or {}).get("query_ground_truth_data_path")
    )
    candidate = requested_path or manifest_clean_path
    if not candidate:
        return (os.path.realpath(str(observation_data_path)), "observation_data_path")
    raw_path = os.path.expanduser(str(candidate))
    candidates = [raw_path]
    if data_manifest_path:
        candidates.append(
            os.path.join(
                os.path.dirname(os.path.abspath(data_manifest_path)),
                os.path.basename(raw_path),
            )
        )
    for path in candidates:
        if os.path.isfile(path):
            source = "explicit_cli" if requested_path else "manifest_clean_data_path"
            return (os.path.realpath(path), source)
    raise FileNotFoundError(
        f"Evaluation ground-truth data path does not exist: {candidate!s} (manifest={data_manifest_path})"
    )


def _load_evaluation_ground_truth_data(
    *,
    ground_truth_path,
    observation_data_path,
    observation_data,
    norm_stats,
    spatial_dim,
):
    """Load and normalize clean GT without changing the noisy inference tensor."""
    if os.path.realpath(str(ground_truth_path)) == os.path.realpath(
        str(observation_data_path)
    ):
        return observation_data
    clean_extract = np.array(
        _load_npy_flexible(ground_truth_path), dtype=np.float32, copy=True
    )
    clean_norm = apply_saved_normalization(clean_extract, norm_stats)
    clean_data = _ensure_5d(clean_norm, spatial_dim=spatial_dim)
    if tuple((int(value) for value in clean_data.shape)) != tuple(
        (int(value) for value in observation_data.shape)
    ):
        raise ValueError(
            f"Clean evaluation GT shape does not match noisy observation data: gt={tuple(clean_data.shape)} observations={tuple(observation_data.shape)} path={ground_truth_path}"
        )
    return clean_data


def main():
    args = _parse_args()
    device = torch.device(
        args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    )
    results_dir = _build_results_dir(args.results_dir, prefix=args.results_prefix)
    log_path = os.path.join(results_dir, "run.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = open(log_path, "a", encoding="utf-8")
    sys.stdout = _TeeStdout(original_stdout, log_handle)
    sys.stderr = _TeeStdout(original_stderr, log_handle)
    previous_query_chunk_size = None
    query_chunk_size_active = False
    try:
        print(f"log path: {log_path}")
        print(f"cwd: {os.getcwd()}")
        print(f"argv: {' '.join(sys.argv)}")
        print("[stage] resolving checkpoint paths")
        ftm_manifest, basis_path, core_path = _resolve_checkpoint_paths(args)
        ftm_manifest, basis_function, ind_input, _ = load_basis_and_coords(
            args.ftm_manifest_path, device
        )
        previous_query_chunk_size = _set_active_pde_query_chunk_size(
            basis_function, int(args.pde_query_chunk_size)
        )
        query_chunk_size_active = True
        basis_path = ftm_manifest["basis_path"]
        core_path = ftm_manifest["core_path"]
        artifact_parameterization = str(
            ftm_manifest.get(
                "core_parameterization",
                ftm_manifest.get("model", {}).get("core_parameterization", "per_time"),
            )
        ).lower()
        if artifact_parameterization != "per_time" or hasattr(basis_function, "mode_t"):
            raise ValueError(
                f"reconstruct_temporal.py supports only per-time RKHS artifacts; got core_parameterization={artifact_parameterization!r}."
            )
        print("[stage] loading saved core payload")
        core_payload = load_core_payload(core_path, ftm_manifest=ftm_manifest)
        norm_stats = core_payload["norm_stats"]
        core_train_np = _as_core_train_array(core_payload)
        data_manifest_path = args.data_manifest_path
        if ftm_manifest is not None and (not data_manifest_path):
            data_manifest_path = str(ftm_manifest.get("source_data_manifest_path", ""))
        if not data_manifest_path:
            raise ValueError(
                "data.data_manifest_path is unset and the RKHS artifact does not record source_data_manifest_path."
            )
        args.data_manifest_path = data_manifest_path
        args.device = str(device)
        args.results_dir = results_dir
        save_resolved_temporal_inference_config(
            args, os.path.join(results_dir, "resolved_config.yaml")
        )
        with open(data_manifest_path, "r", encoding="utf-8") as manifest_handle:
            manifest_preview = json.load(manifest_handle)
        validation_split = "val"
        if "val" not in manifest_preview.get("splits", {}):
            requested_model_path = str(args.temporal_model_path or "")
            requested_model_best_path = (
                os.path.splitext(requested_model_path)[0] + "_best.pth"
                if requested_model_path
                else ""
            )
            odernn_checkpoint_ready = False
            odernn_checkpoint_ready = bool(
                int(args.temporal_load_if_exists) == 1
                and requested_model_path
                and (
                    os.path.isfile(requested_model_path)
                    or os.path.isfile(requested_model_best_path)
                )
            )
            can_reuse_test_split = "odernn" in set() or odernn_checkpoint_ready
            if can_reuse_test_split and "test" in manifest_preview.get("splits", {}):
                validation_split = "test"
                print(
                    "[stage] validation split is absent; using test split for temporal inference setup"
                )
            else:
                raise KeyError(
                    f"Data manifest has no val split and temporal_method={'odernn'!r} requires one."
                )
        print(f"[stage] resolving split paths | split={validation_split}")
        data_path, metadata_path, data_manifest = resolve_split_paths(
            data_manifest_path=data_manifest_path, split=validation_split
        )
        print("[stage] loading validation split data")
        ind_uni, data_extract, mask_np = _load_large_data_flexible(
            data_path, metadata_path
        )
        print("[stage] applying saved normalization")
        data_norm = apply_saved_normalization(data_extract, norm_stats)
        spatial_dim = 3 if hasattr(basis_function, "mode_w") else 2
        data = _ensure_5d(data_norm, spatial_dim=spatial_dim)
        _validate_per_time_split(ind_uni, data, mask_np, ftm_manifest, "validation")
        num_channels = int(data.shape[-1])
        save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
        shared_basis_multichannel = (
            bool(core_payload.get("shared_basis_multichannel", False))
            or num_channels > 1
        )
        _check_core_compatible(
            core_train_np,
            data,
            shared_basis_multichannel,
            core_parameterization="per_time",
        )
        channel_names = _infer_channel_names(data_manifest, num_channels)
        num_samples = int(data.shape[0])
        nt = int(data.shape[1])
        batch_size = max(int(args.batch_size), 1)
        obs_ratio_list = _parse_float_list_csv(
            args.temporal_obs_ratio_list, "--temporal_obs_ratio_list"
        )
        time_indices = _parse_plot_time_indices(args.plot_time_indices, nt)
        time_values = _artifact_time_values(ftm_manifest, ind_uni[3])
        query_time_values = _parse_query_times(args, time_values)
        query_matches_eval_grid = _times_same_grid(
            query_time_values, time_values, tol=float(args.query_time_match_tol)
        )
        inference_time_keep = _build_inference_time_mask(
            num_times=nt,
            template=str(args.inference_time_template),
            seed=int(args.inference_time_seed),
        )
        inference_mask_np = _apply_time_keep_to_mask(mask_np, inference_time_keep)
        inference_obs_ratio = _resolve_inference_obs_ratio(
            inference_mask_np, float(args.inference_obs_ratio)
        )
        inference_mask_rng = np.random.default_rng(int(args.inference_mask_seed))
        inference_mask_np = _apply_time_mask_mode(
            inference_mask_np,
            inference_obs_ratio,
            inference_mask_rng,
            str(args.inference_time_mask_mode),
            args,
        )
        obs_time_mask_original = _time_observation_mask(mask_np)
        obs_time_mask = _time_observation_mask(inference_mask_np)
        print("[stage] loading basis function")
        basis_function.eval()
        if hasattr(basis_function, "mode_w"):
            rank = (
                int(basis_function.r_u),
                int(basis_function.r_v),
                int(basis_function.r_w),
            )
        else:
            rank = (
                int(getattr(basis_function, "r_1")),
                int(getattr(basis_function, "r_2")),
                1,
            )
        with torch.no_grad():
            basis_function.mode = "training"
            precomputed_basis = basis_function(input_ind_train=ind_input)
        print(f"device: {device}")
        print(f"basis path: {basis_path}")
        print(f"core path: {core_path}")
        print(f"results dir: {results_dir}")
        print(f"data path: {data_path}")
        print(f"data shape: {tuple(data.shape)}")
        print(
            f"shared_basis_multichannel: {shared_basis_multichannel} | channels: {channel_names}"
        )
        print(f"temporal core shape: {tuple(core_train_np.shape)}")
        print(f"temporal_method: {str('odernn').lower()}")
        print(f"temporal_model_space: {str('core').lower()}")
        print(f"rank: {rank}")
        temporal_dataset_type = _resolve_temporal_pde_dataset_type(
            args.pde_dataset_type,
            data_manifest_path=data_manifest_path,
            data_manifest=data_manifest,
            ftm_manifest=ftm_manifest,
            num_channels=num_channels,
        )
        pde_kwargs = (
            None
            if temporal_dataset_type is None
            else _build_temporal_pde_kwargs(args, temporal_dataset_type)
        )
        if pde_kwargs is not None and temporal_dataset_type == "heat3d":
            pde_kwargs["time_coords_physical"] = _load_physical_time_coords(
                metadata_path, time_values
            )
        pde_channel_names = (
            None
            if temporal_dataset_type is None
            else pde_channel_order(temporal_dataset_type, num_channels=num_channels)
        )
        if pde_kwargs:
            print(
                "temporal PDE setup:",
                f"dataset_type={temporal_dataset_type}",
                f"requested={str(args.pde_dataset_type).lower()}",
                f"pde_start_step={int(args.pde_start_step)}",
                f"monitor_always={bool(args.pde_monitor_always)}",
                f"monitor_point={bool(args.pde_monitor_point)}",
                f"kwargs={pde_kwargs}",
            )
        else:
            print("temporal PDE setup: disabled (pde_dataset_type=none)")
        print(
            "validation time mask:",
            f"template={args.inference_time_template}",
            (
                f"drop_ratio={float(args.inference_time_drop_ratio):.4f}"
                if str(args.inference_time_template).lower() == "random_drop"
                else "drop_ratio=n/a"
            ),
            f"time_mask_mode={str(args.inference_time_mask_mode).lower()}",
            f"time_keep_ratio={float(inference_obs_ratio):.4f}",
            f"template_kept={int(np.sum(inference_time_keep))}/{nt}",
            f"final_kept_mean={float(np.mean(np.sum(obs_time_mask, axis=1))):.2f}/{nt}",
            f"original_obs_time_ratio={float(np.mean(obs_time_mask_original)):.4f}",
            f"inference_obs_time_ratio={float(np.mean(obs_time_mask)):.4f}",
        )
        val_data = data
        val_mask_np = mask_np
        validation_ind_uni = ind_uni
        validation_data_path = data_path
        validation_metadata_path = metadata_path
        validation_data_manifest = data_manifest
        temporal_model_path = args.temporal_model_path or os.path.join(
            results_dir, "temporal_residual_odernn.pth"
        )
        if args.temporal_save_model:
            os.makedirs(
                os.path.dirname(os.path.abspath(temporal_model_path)), exist_ok=True
            )
        temporal_model_best_path = (
            os.path.splitext(temporal_model_path)[0] + "_best.pth"
        )
        load_existing_model = os.path.exists(
            temporal_model_best_path
        ) or os.path.exists(temporal_model_path)
        train_data_norm = None
        train_data_t = None
        train_mask_np = None
        mean_core_np = np.mean(core_train_np, axis=0, dtype=np.float64).astype(
            np.float32
        )
        load_train_split_for_temporal = "odernn" == "odernn" and (
            not (int(args.temporal_load_if_exists) == 1 and load_existing_model)
        )
        if load_train_split_for_temporal or False:
            print("[stage] loading train split full field")
            train_data_path, train_metadata_path, _ = resolve_split_paths(
                data_manifest_path=data_manifest_path, split="train"
            )
            train_ind_uni, train_data_extract, train_mask_np = (
                _load_large_data_flexible(train_data_path, train_metadata_path)
            )
            train_data_norm = np.asarray(
                apply_saved_normalization(train_data_extract, norm_stats),
                dtype=np.float32,
            )
            if train_data_norm.ndim == 4:
                train_data_norm = train_data_norm[..., None]
            train_data_t = _ensure_5d(train_data_norm, spatial_dim=spatial_dim)
            train_mask_np = np.asarray(train_mask_np, dtype=np.float32)
            _validate_per_time_split(
                train_ind_uni, train_data_t, train_mask_np, ftm_manifest, "train"
            )
            if int(train_data_t.shape[0]) != int(core_train_np.shape[0]):
                raise ValueError(
                    f"Train data samples {int(train_data_t.shape[0])} != saved train core samples {int(core_train_np.shape[0])}"
                )
        temporal_model = None
        temporal_model_arch = TEMPORAL_MODEL_ARCH
        core_dim = _series_feature_dim(tuple(core_train_np.shape))
        temporal_model_kwargs = {
            "core_dim": core_dim,
            "hidden_dim": int(args.temporal_hidden_dim),
            "ode_hidden_dim": int(args.temporal_ode_hidden_dim),
        }
        if int(args.temporal_load_if_exists) == 1 and load_existing_model:
            load_path = (
                temporal_model_best_path
                if os.path.exists(temporal_model_best_path)
                else temporal_model_path
            )
            print("[stage] loading temporal residual model checkpoint")
            payload = torch.load(load_path, map_location=device, weights_only=False)
            checkpoint_arch = _detect_temporal_checkpoint_arch(payload)
            temporal_model_arch = checkpoint_arch
            _validate_temporal_checkpoint(
                payload,
                checkpoint_arch,
                core_dim=core_dim,
                args=args,
                time_values=time_values,
                model_space="core",
                load_path=load_path,
            )
            model_class = _temporal_model_class_for_checkpoint_arch(checkpoint_arch)
            if checkpoint_arch != TEMPORAL_MODEL_ARCH:
                print(
                    "[checkpoint] loading pre-Hermite temporal architecture with matching implementation:",
                    checkpoint_arch,
                )
            temporal_model = model_class(**temporal_model_kwargs).to(device)
            try:
                temporal_model.load_state_dict(payload["model_state"], strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    f"Temporal checkpoint {load_path} is incompatible with arch={checkpoint_arch}. Retrain the temporal model with this script before loading it."
                ) from exc
            print(f"loaded temporal residual model: {load_path} arch={checkpoint_arch}")
        else:
            temporal_model = TemporalResidualODERNN(**temporal_model_kwargs).to(device)
            print(
                "temporal residual training:",
                f"arch={TEMPORAL_MODEL_ARCH}",
                f"epochs={int(args.temporal_train_epochs)}",
                f"hidden={int(args.temporal_hidden_dim)}",
                f"train_time_modes={','.join(_temporal_time_mask_modes(args, kind='train'))}",
                f"val_time_modes={','.join(_temporal_time_mask_modes(args, kind='val'))}",
                f"val_samples={int(val_data.shape[0])}",
                f"obs_ratio_list={obs_ratio_list}",
            )
            temporal_model = _train_temporal_residual_model(
                model=temporal_model,
                core_train_np=core_train_np,
                train_data_t=train_data_t,
                train_mask_np=train_mask_np,
                obs_ratio_list=obs_ratio_list,
                time_values=time_values,
                val_data_t=val_data,
                val_mask_np=val_mask_np,
                mean_core_np=mean_core_np,
                train_data_norm=train_data_norm,
                precomputed_basis=precomputed_basis,
                basis_function=basis_function,
                ind_input=ind_input,
                rank=rank,
                norm_stats=norm_stats,
                shared_basis_multichannel=shared_basis_multichannel,
                args=args,
                device=device,
                coord_uniques=ind_input,
                dataset_type=(
                    "none" if temporal_dataset_type is None else temporal_dataset_type
                ),
                pde_kwargs=pde_kwargs,
                pde_channel_order_names=pde_channel_names,
                experiment_root_dir=results_dir,
                best_model_path=temporal_model_best_path,
                temporal_model_space="core",
            )
            if int(args.temporal_save_model) == 1:
                torch.save(
                    _temporal_checkpoint_payload(
                        temporal_model, args, time_values, "core"
                    ),
                    temporal_model_path,
                )
                print(f"saved temporal residual model: {temporal_model_path}")
                if os.path.exists(temporal_model_best_path):
                    print(
                        f"saved best temporal residual model: {temporal_model_best_path}"
                    )
        if int(args.do_inference) != 1:
            print(
                "[stage] temporal setup finished | do_inference=0, skip test inference"
            )
            return
        if validation_split == "test":
            print("[stage] reusing test-only split loaded during temporal setup")
            data_path = validation_data_path
            metadata_path = validation_metadata_path
            data_manifest = validation_data_manifest
            ind_uni = validation_ind_uni
            data = val_data
            mask_np = val_mask_np
        else:
            print("[stage] resolving split paths | split=test")
            data_path, metadata_path, data_manifest = resolve_split_paths(
                data_manifest_path=data_manifest_path, split="test"
            )
            print("[stage] loading test split data")
            ind_uni, data_extract, mask_np = _load_large_data_flexible(
                data_path, metadata_path
            )
            print("[stage] applying saved normalization to test split")
            data_norm = apply_saved_normalization(data_extract, norm_stats)
            data = _ensure_5d(data_norm, spatial_dim=spatial_dim)
            _validate_per_time_split(ind_uni, data, mask_np, ftm_manifest, "test")
        evaluation_ground_truth_data_path, evaluation_ground_truth_source = (
            _resolve_evaluation_ground_truth_path(
                requested_path=getattr(args, "evaluation_ground_truth_data_path", None),
                data_manifest_path=data_manifest_path,
                data_manifest=data_manifest,
                observation_data_path=data_path,
            )
        )
        evaluation_data = _load_evaluation_ground_truth_data(
            ground_truth_path=evaluation_ground_truth_data_path,
            observation_data_path=data_path,
            observation_data=data,
            norm_stats=norm_stats,
            spatial_dim=spatial_dim,
        )
        print(f"[stage] evaluation GT path: {evaluation_ground_truth_data_path}")
        print(f"[stage] evaluation GT source: {evaluation_ground_truth_source}")
        print(f"[stage] evaluation GT shape: {tuple(evaluation_data.shape)}")
        test_pde_kwargs = None if pde_kwargs is None else dict(pde_kwargs)
        if spatial_dim == 3:
            spatial_order = spatial_axis_order_from_manifest(
                ftm_manifest, spatial_dim=3
            )
            test_coord_uniques = ordered_spatial_values(
                spatial_order,
                u=torch.as_tensor(ind_uni[0], dtype=torch.float32, device=device),
                v=torch.as_tensor(ind_uni[1], dtype=torch.float32, device=device),
                w=torch.as_tensor(ind_uni[2], dtype=torch.float32, device=device),
            )
        else:
            test_coord_uniques = (
                torch.as_tensor(ind_uni[0], dtype=torch.float32, device=device),
                torch.as_tensor(ind_uni[1], dtype=torch.float32, device=device),
            )
        num_samples = int(data.shape[0])
        nt = int(data.shape[1])
        time_indices = _parse_plot_time_indices(args.plot_time_indices, nt)
        time_values = _artifact_time_values(ftm_manifest, ind_uni[3])
        if temporal_dataset_type == "heat3d" and test_pde_kwargs is not None:
            test_pde_kwargs["time_coords_physical"] = _load_physical_time_coords(
                metadata_path, time_values
            )
        query_time_values = _parse_query_times(args, time_values)
        query_matches_eval_grid = _times_same_grid(
            query_time_values, time_values, tol=float(args.query_time_match_tol)
        )
        pde_time_values = np.asarray(time_values, dtype=np.float32).copy()
        pde_physical_time_values = _load_physical_time_coords(
            metadata_path, time_values
        )
        pde_source_time_dt = _infer_time_step(pde_physical_time_values)
        pde_query_time_dt = pde_source_time_dt
        pde_time_query_enabled = False
        pde_query_matches_eval_grid = True
        pde_query_matches_user_query = _times_same_grid(
            pde_time_values, query_time_values, tol=float(args.query_time_match_tol)
        )
        test_pde_eval_kwargs = None
        if test_pde_kwargs:
            test_pde_eval_kwargs = dict(test_pde_kwargs)
            if temporal_dataset_type in ("ac", "ns"):
                test_pde_eval_kwargs["dt"] = float(pde_source_time_dt)
            if temporal_dataset_type == "heat3d":
                test_pde_eval_kwargs["time_coords_physical"] = np.asarray(
                    pde_physical_time_values, dtype=np.float32
                ).copy()
        inference_time_keep = _build_inference_time_mask(
            num_times=nt,
            template=str(args.inference_time_template),
            seed=int(args.inference_time_seed),
        )
        inference_mask_np = _apply_time_keep_to_mask(mask_np, inference_time_keep)
        inference_obs_ratio = _resolve_inference_obs_ratio(
            inference_mask_np, float(args.inference_obs_ratio)
        )
        inference_mask_rng = np.random.default_rng(int(args.inference_mask_seed))
        inference_mask_np = _apply_time_mask_mode(
            inference_mask_np,
            inference_obs_ratio,
            inference_mask_rng,
            str(args.inference_time_mask_mode),
            args,
        )
        obs_time_mask_original = _time_observation_mask(mask_np)
        obs_time_mask = _time_observation_mask(inference_mask_np)
        print(f"[stage] test data path: {data_path}")
        print(f"[stage] test data shape: {tuple(data.shape)}")
        print(
            "[stage] test inference time mask:",
            f"template={args.inference_time_template}",
            (
                f"drop_ratio={float(args.inference_time_drop_ratio):.4f}"
                if str(args.inference_time_template).lower() == "random_drop"
                else "drop_ratio=n/a"
            ),
            f"time_mask_mode={str(args.inference_time_mask_mode).lower()}",
            f"time_keep_ratio={float(inference_obs_ratio):.4f}",
            f"template_kept={int(np.sum(inference_time_keep))}/{nt}",
            f"final_kept_mean={float(np.mean(np.sum(obs_time_mask, axis=1))):.2f}/{nt}",
            f"original_obs_time_ratio={float(np.mean(obs_time_mask_original)):.4f}",
            f"inference_obs_time_ratio={float(np.mean(obs_time_mask)):.4f}",
        )
        if test_pde_eval_kwargs:
            print(
                "temporal PDE time query:",
                f"enabled={bool(pde_time_query_enabled)}",
                f"substeps={max(int(args.pde_time_query_substeps), 1)}",
                f"source_dt={float(pde_source_time_dt):.6g}",
                f"query_dt={float(pde_query_time_dt):.6g}",
                f"query_steps={int(pde_time_values.size)}",
                f"physical_dt={(None if test_pde_eval_kwargs.get('dt') is None else float(test_pde_eval_kwargs['dt']))}",
            )
        inference_timings = {}
        _sync_device_if_needed(device)
        anchor_prepare_t0 = time.perf_counter()
        print("[stage] solving observed-time anchor cores")
        anchor_core_np, anchor_prior_np = _solve_anchor_cores_for_dataset(
            data_t=data,
            mask_np=inference_mask_np,
            core_train_np=core_train_np,
            mean_core_np=mean_core_np,
            train_data_norm=train_data_norm,
            train_mask_np=train_mask_np,
            precomputed_basis=precomputed_basis,
            basis_function=basis_function,
            ind_input=ind_input,
            rank=rank,
            shared_basis_multichannel=shared_basis_multichannel,
            args=args,
            device=device,
            batch_size=batch_size,
            log_prefix="[anchor solve]",
        )
        anchor_series_np = anchor_core_np
        _sync_device_if_needed(device)
        inference_timings["anchor_prepare_seconds"] = float(
            time.perf_counter() - anchor_prepare_t0
        )
        print(f"[stage] running temporal {'odernn'} head on eval grid")
        _sync_device_if_needed(device)
        temporal_head_t0 = time.perf_counter()
        (
            baseline_eval_np,
            gate_eval_np,
            residual_eval_np,
            core_final_np,
            eval_matched_anchor_count,
        ) = _run_temporal_head_on_grid(
            temporal_method="odernn",
            temporal_model=temporal_model,
            anchor_series_np=anchor_series_np,
            source_times_np=time_values,
            obs_time_mask_np=obs_time_mask,
            target_times_np=time_values,
            args=args,
            device=device,
        )
        if query_matches_eval_grid:
            baseline_query_np = baseline_eval_np
            gate_query_np = gate_eval_np
            residual_query_np = residual_eval_np
            core_query_final_np = core_final_np
            query_matched_anchor_count = eval_matched_anchor_count
        else:
            print(f"[stage] running temporal {'odernn'} head on query grid")
            (
                baseline_query_np,
                gate_query_np,
                residual_query_np,
                core_query_final_np,
                query_matched_anchor_count,
            ) = _run_temporal_head_on_grid(
                temporal_method="odernn",
                temporal_model=temporal_model,
                anchor_series_np=anchor_series_np,
                source_times_np=time_values,
                obs_time_mask_np=obs_time_mask,
                target_times_np=query_time_values,
                args=args,
                device=device,
            )
        _sync_device_if_needed(device)
        inference_timings["temporal_head_forward_seconds"] = float(
            time.perf_counter() - temporal_head_t0
        )
        print("[stage] decoding eval-grid fields and computing metrics")
        _sync_device_if_needed(device)
        decode_eval_t0 = time.perf_counter()
        pred_all, metric_accum = _decode_core_batches(
            core_np=core_final_np,
            precomputed_basis=precomputed_basis,
            norm_stats=norm_stats,
            batch_size=batch_size,
            save_dtype=save_dtype,
            device=device,
            channel_names=channel_names,
            data=evaluation_data,
            mask_np=inference_mask_np,
            results_dir=results_dir,
            time_indices=time_indices,
            n_plot_samples=int(args.n_plot_samples),
            shared_basis_multichannel=shared_basis_multichannel,
            spatial_dim=spatial_dim,
        )
        missing_time_keep = ~np.asarray(obs_time_mask, dtype=bool)
        rel_l2_missing = _compute_time_subset_rel_l2_batched(
            pred_np=pred_all,
            data_t=evaluation_data,
            norm_stats=norm_stats,
            time_mask=missing_time_keep,
            batch_size=batch_size,
            device=device,
        )
        rmse_missing = _compute_time_subset_rmse_batched(
            pred_np=pred_all,
            data_t=evaluation_data,
            norm_stats=norm_stats,
            time_mask=missing_time_keep,
            batch_size=batch_size,
            device=device,
        )
        _sync_device_if_needed(device)
        inference_timings["decode_eval_metrics_seconds"] = float(
            time.perf_counter() - decode_eval_t0
        )
        inference_timings["pre_pde_total_seconds"] = float(
            inference_timings["anchor_prepare_seconds"]
            + inference_timings["temporal_head_forward_seconds"]
            + inference_timings["decode_eval_metrics_seconds"]
        )
        for key, value in list(inference_timings.items()):
            inference_timings[f"{key}_per_sequence"] = float(
                value / max(num_samples, 1)
            )
        temporal_head_scope_text = (
            "temporal_head_forward = ODE-RNN residual model forward on eval/query grid"
        )
        timing_payload = {
            "scope": f"test split only; anchor_prepare = observed-time anchor core solve plus optional anchor decode; {temporal_head_scope_text}; decode_eval_metrics = eval-grid decode plus relL2 metric accumulation; pre_pde_total excludes PDE residual, SSIM, plots, and file saving",
            "num_sequences": int(num_samples),
            "temporal_method": "odernn",
            "anchor_prepare_seconds": float(
                inference_timings["anchor_prepare_seconds"]
            ),
            "anchor_prepare_per_sequence_seconds": float(
                inference_timings["anchor_prepare_seconds_per_sequence"]
            ),
            "temporal_head_forward_seconds": float(
                inference_timings["temporal_head_forward_seconds"]
            ),
            "temporal_head_forward_per_sequence_seconds": float(
                inference_timings["temporal_head_forward_seconds_per_sequence"]
            ),
            "decode_eval_metrics_seconds": float(
                inference_timings["decode_eval_metrics_seconds"]
            ),
            "decode_eval_metrics_per_sequence_seconds": float(
                inference_timings["decode_eval_metrics_seconds_per_sequence"]
            ),
            "pre_pde_total_seconds": float(inference_timings["pre_pde_total_seconds"]),
            "pre_pde_total_per_sequence_seconds": float(
                inference_timings["pre_pde_total_seconds_per_sequence"]
            ),
        }
        timing_path = os.path.join(results_dir, "temporal_inference_times.json")
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(timing_payload, f, indent=2)
            f.write("\n")
        print(f"saved temporal inference times: {timing_path}")
        print(
            "temporal inference timing:",
            f"anchor={timing_payload['anchor_prepare_per_sequence_seconds']:.6f}s/seq",
            f"{'odernn'}={timing_payload['temporal_head_forward_per_sequence_seconds']:.6f}s/seq",
            f"decode={timing_payload['decode_eval_metrics_per_sequence_seconds']:.6f}s/seq",
            f"pre_pde_total={timing_payload['pre_pde_total_per_sequence_seconds']:.6f}s/seq",
        )
        pde_series_final_np = None
        pde_query_matched_anchor_count = 0
        if test_pde_eval_kwargs:
            if pde_query_matches_eval_grid:
                pde_series_final_np = core_final_np
                pde_query_matched_anchor_count = eval_matched_anchor_count
            elif not query_matches_eval_grid and pde_query_matches_user_query:
                pde_series_final_np = core_query_final_np
                pde_query_matched_anchor_count = query_matched_anchor_count
            else:
                print(f"[stage] running temporal {'odernn'} head on PDE time grid")
                (
                    _baseline_pde_np,
                    _gate_pde_np,
                    _residual_pde_np,
                    pde_series_final_np,
                    pde_query_matched_anchor_count,
                ) = _run_temporal_head_on_grid(
                    temporal_method="odernn",
                    temporal_model=temporal_model,
                    anchor_series_np=anchor_series_np,
                    source_times_np=time_values,
                    obs_time_mask_np=obs_time_mask,
                    target_times_np=pde_time_values,
                    args=args,
                    device=device,
                )
        test_pde_metrics = {}
        if test_pde_eval_kwargs:
            use_direct_pde_query = (
                "core" != "field"
                and (not (temporal_dataset_type == "ns" and pde_channel_names is None))
                and _can_use_ftm_continuous_pde_query(
                    pde_series_final_np,
                    basis_function,
                    shared_basis_multichannel=shared_basis_multichannel,
                )
            )
            if temporal_dataset_type == "ns" and pde_channel_names is None:
                pred_pde_np = _decode_core_only(
                    core_np=pde_series_final_np,
                    precomputed_basis=precomputed_basis,
                    norm_stats=norm_stats,
                    batch_size=batch_size,
                    save_dtype=save_dtype,
                    device=device,
                    num_channels=num_channels,
                    shared_basis_multichannel=shared_basis_multichannel,
                )
                pred_pde_time_values = pde_time_values
                pred_pde_core_np = None
            elif use_direct_pde_query:
                pred_pde_np = pred_all
                pred_pde_time_values = pde_time_values
                pred_pde_core_np = pde_series_final_np
            elif pde_query_matches_eval_grid:
                pred_pde_np = pred_all
                pred_pde_time_values = time_values
                pred_pde_core_np = None
            else:
                pred_pde_np = _decode_core_only(
                    core_np=pde_series_final_np,
                    precomputed_basis=precomputed_basis,
                    norm_stats=norm_stats,
                    batch_size=batch_size,
                    save_dtype=save_dtype,
                    device=device,
                    num_channels=num_channels,
                    shared_basis_multichannel=shared_basis_multichannel,
                )
                pred_pde_time_values = pde_time_values
                pred_pde_core_np = None
            test_pde_metrics = _compute_temporal_pde_batched(
                pred_np=pred_pde_np,
                data_t=evaluation_data,
                norm_stats=norm_stats,
                time_mask=np.ones(int(pde_time_values.size), dtype=bool),
                pred_time_values_np=pred_pde_time_values,
                gt_time_values_np=time_values,
                target_time_values_np=pde_time_values,
                batch_size=batch_size,
                device=device,
                coord_uniques=test_coord_uniques,
                dataset_type=temporal_dataset_type,
                pde_kwargs=test_pde_eval_kwargs,
                channel_order=pde_channel_names,
                point_sample_ratio=float(args.pde_point_sample_ratio),
                point_max_points=int(args.pde_point_max_points),
                pred_core_np=pred_pde_core_np,
                basis_function=basis_function,
                shared_basis_multichannel=shared_basis_multichannel,
            )
            if bool(int(args.plot_pde_residual_on_log)) == 1:
                _save_temporal_pde_snapshot(
                    pred_np=pred_pde_np,
                    data_t=evaluation_data,
                    norm_stats=norm_stats,
                    pred_time_values_np=pred_pde_time_values,
                    gt_time_values_np=time_values,
                    target_time_values_np=pde_time_values,
                    coord_uniques=test_coord_uniques,
                    dataset_type=temporal_dataset_type,
                    pde_kwargs=test_pde_eval_kwargs,
                    channel_order=pde_channel_names,
                    point_sample_ratio=float(args.pde_point_sample_ratio),
                    point_max_points=int(args.pde_point_max_points),
                    results_dir=results_dir,
                    step_tag="test_inference",
                    sample_indices=_parse_pde_snapshot_sample_indices(args),
                    device=device,
                    pred_core_np=pred_pde_core_np,
                    basis_function=basis_function,
                    precomputed_basis=precomputed_basis,
                    shared_basis_multichannel=shared_basis_multichannel,
                )
        print("[stage] computing temporal protocol sensor/off-sensor metrics")
        temporal_protocol_metrics = _compute_temporal_protocol_metrics_batched(
            predictions={"": pred_all},
            data_t=evaluation_data,
            norm_stats=norm_stats,
            mask_np=inference_mask_np,
            batch_size=batch_size,
            device=device,
        )
        pred_query = None
        if not query_matches_eval_grid:
            print("[stage] decoding query-grid fields")
            pred_query = _decode_core_only(
                core_np=core_query_final_np,
                precomputed_basis=precomputed_basis,
                norm_stats=norm_stats,
                batch_size=batch_size,
                save_dtype=save_dtype,
                device=device,
                num_channels=num_channels,
                shared_basis_multichannel=shared_basis_multichannel,
            )
        channel_metrics = _finalize_channel_metrics(metric_accum, channel_names)
        total_sse_full = float(sum(metric_accum["sse_full"]))
        total_sae_full = float(sum(metric_accum["sae_full"]))
        total_sgt2_full = float(sum(metric_accum["sgt2_full"]))
        total_count_full = int(sum(metric_accum["count_full"]))
        total_sse_obs = float(sum(metric_accum["sse_obs"]))
        total_sae_obs = float(sum(metric_accum["sae_obs"]))
        total_sgt2_obs = float(sum(metric_accum["sgt2_obs"]))
        total_count_obs = int(sum(metric_accum["count_obs"]))
        mse_full = float(total_sse_full / max(total_count_full, 1))
        rmse_full = float(np.sqrt(mse_full))
        mae_full = float(total_sae_full / max(total_count_full, 1))
        rel_l2_full = float(np.sqrt(total_sse_full / max(total_sgt2_full, 1e-12)))
        mse_obs = (
            float(total_sse_obs / max(total_count_obs, 1))
            if total_count_obs > 0
            else float("nan")
        )
        rmse_obs = float(np.sqrt(mse_obs)) if total_count_obs > 0 else float("nan")
        mae_obs = (
            float(total_sae_obs / max(total_count_obs, 1))
            if total_count_obs > 0
            else float("nan")
        )
        rel_l2_obs = (
            float(np.sqrt(total_sse_obs / max(total_sgt2_obs, 1e-12)))
            if total_count_obs > 0
            else float("nan")
        )
        total_sse_nonsensor = max(total_sse_full - total_sse_obs, 0.0)
        total_sae_nonsensor = max(total_sae_full - total_sae_obs, 0.0)
        total_sgt2_nonsensor = max(total_sgt2_full - total_sgt2_obs, 0.0)
        total_count_nonsensor = max(total_count_full - total_count_obs, 0)
        mse_nonsensor = (
            float(total_sse_nonsensor / max(total_count_nonsensor, 1))
            if total_count_nonsensor > 0
            else float("nan")
        )
        rmse_nonsensor = (
            float(np.sqrt(mse_nonsensor)) if total_count_nonsensor > 0 else float("nan")
        )
        mae_nonsensor = (
            float(total_sae_nonsensor / max(total_count_nonsensor, 1))
            if total_count_nonsensor > 0
            else float("nan")
        )
        rel_l2_nonsensor = (
            float(np.sqrt(total_sse_nonsensor / max(total_sgt2_nonsensor, 1e-12)))
            if total_count_nonsensor > 0
            else float("nan")
        )
        gap_nonsensor_sensor = (
            float(rel_l2_nonsensor / rel_l2_obs)
            if total_count_nonsensor > 0
            and total_count_obs > 0
            and np.isfinite(rel_l2_obs)
            and (abs(rel_l2_obs) > 1e-12)
            else float("nan")
        )
        extra_metrics = {}
        if int(args.skip_ssim) == 1:
            print("[stage] skipping SSIM metrics")
        else:
            pred_for_extra = (
                pred_all[..., None]
                if num_channels == 1 and pred_all.ndim == 4
                else pred_all
            )
            extra_metrics = compute_ssim_metrics_batched(
                pred_np=pred_for_extra,
                data_t=evaluation_data,
                norm_stats=norm_stats,
                mask_np=inference_mask_np,
                batch_size=batch_size,
            )
        print("[stage] saving predictions and metrics")
        pred_save_path = os.path.join(results_dir, "predictions_physical.npz")
        save_payload = {
            "pred": pred_all,
            "core_anchor": anchor_core_np.astype(np.float32),
            "core_anchor_prior": anchor_prior_np.astype(np.float32),
            "gate": gate_eval_np.astype(np.float32),
            "obs_time_mask_original": obs_time_mask_original.astype(np.uint8),
            "obs_time_mask": obs_time_mask.astype(np.uint8),
            "inference_time_keep": inference_time_keep.astype(np.uint8),
            "inference_time_mask_mode": np.asarray(
                str(args.inference_time_mask_mode).lower()
            ),
            "inference_time_keep_ratio": np.asarray(
                float(inference_obs_ratio), dtype=np.float32
            ),
            "t": time_values.astype(np.float32),
            "eval_times": time_values.astype(np.float32),
            "query_times": query_time_values.astype(np.float32),
            "pde_query_times": pde_time_values.astype(np.float32),
            "u": np.asarray(ind_uni[0], dtype=np.float32),
            "v": np.asarray(ind_uni[1], dtype=np.float32),
            "w": (
                np.asarray(ind_uni[2], dtype=np.float32)
                if spatial_dim == 3
                else np.asarray([0.0], dtype=np.float32)
            ),
            "spatial_axis_order": np.asarray(
                spatial_order if spatial_dim == 3 else ("u", "v"), dtype=object
            ),
            "channel_names": np.asarray(channel_names, dtype=object),
            "temporal_method": np.asarray("odernn"),
            "temporal_model_space": np.asarray("core"),
        }
        for spectral_key in (
            "psd_radial_bin_indices",
            "psd_gt_radial",
            "psd_pred_radial",
            "psd_abs_error_radial",
            "psd_rel_error_radial",
            "psd_log_error_radial",
        ):
            if spectral_key in extra_metrics:
                dtype = (
                    np.int32 if spectral_key == "psd_radial_bin_indices" else np.float32
                )
                save_payload[spectral_key] = np.asarray(
                    extra_metrics[spectral_key], dtype=dtype
                )
        save_payload.update(
            {
                "core_baseline": baseline_eval_np.astype(np.float32),
                "core_residual": residual_eval_np.astype(np.float32),
                "core_final": core_final_np.astype(np.float32),
            }
        )
        if not query_matches_eval_grid:
            query_payload = {
                "pred_query": pred_query,
                "gate_query": gate_query_np.astype(np.float32),
            }
            query_payload.update(
                {
                    "core_query_baseline": baseline_query_np.astype(np.float32),
                    "core_query_residual": residual_query_np.astype(np.float32),
                    "core_query_final": core_query_final_np.astype(np.float32),
                }
            )
            save_payload.update(query_payload)
        if int(args.skip_save_predictions) == 1:
            print(f"[stage] skipping prediction archive: {pred_save_path}")
        else:
            np.savez_compressed(pred_save_path, **save_payload)
        metrics = {
            "split": "test",
            "num_samples": num_samples,
            "rank": list(rank),
            "num_channels": num_channels,
            "channel_names": channel_names,
            "prior_core_mode": str("mean"),
            "lambda_core_ridge": float(args.lambda_core_ridge),
            "beta_core_tv": float(args.beta_core_tv),
            "cg_max_iter": int(args.cg_max_iter),
            "cg_tol": float(args.cg_tol),
            "temporal_method": "odernn",
            "temporal_hidden_dim": int(args.temporal_hidden_dim),
            "temporal_ode_hidden_dim": int(args.temporal_ode_hidden_dim),
            "temporal_model_arch": temporal_model_arch,
            "temporal_model_space": "core",
            "temporal_train_epochs": int(args.temporal_train_epochs),
            "temporal_train_batch_size": int(args.temporal_train_batch_size),
            "temporal_train_lr": float(args.temporal_train_lr),
            "temporal_train_time_mask_mechanisms": _temporal_time_mask_modes(
                args, kind="train"
            ),
            "temporal_val_time_mask_mechanisms": _temporal_time_mask_modes(
                args, kind="val"
            ),
            "temporal_train_keep_ratio": float(args.temporal_train_keep_ratio),
            "temporal_obs_ratio_list": [float(x) for x in obs_ratio_list],
            "temporal_core_loss_weight": float(args.temporal_core_loss_weight),
            "lambda_pde_point": float(args.lambda_pde_point),
            "pde_start_step": int(args.pde_start_step),
            "pde_monitor_always": bool(args.pde_monitor_always),
            "pde_monitor_point": bool(args.pde_monitor_point),
            "skip_ssim": bool(int(args.skip_ssim)),
            "skip_save_predictions": bool(int(args.skip_save_predictions)),
            "pde_dataset_type": temporal_dataset_type,
            "pde_point_sample_ratio": float(args.pde_point_sample_ratio),
            "pde_point_max_points": int(args.pde_point_max_points),
            "pde_query_chunk_size": int(args.pde_query_chunk_size),
            "pde_kwargs": (
                test_pde_eval_kwargs if int(args.do_inference) == 1 else pde_kwargs
            ),
            "pde_kwargs_base": (
                test_pde_kwargs if int(args.do_inference) == 1 else pde_kwargs
            ),
            "override_anchor_times": bool(args.override_anchor_times),
            "query_num": int(query_time_values.size),
            "query_times_match_eval_grid": bool(query_matches_eval_grid),
            "query_time_start": (
                float(query_time_values[0]) if query_time_values.size > 0 else None
            ),
            "query_time_end": (
                float(query_time_values[-1]) if query_time_values.size > 0 else None
            ),
            "query_matched_anchor_time_count": int(query_matched_anchor_count),
            "eval_matched_anchor_time_count": int(eval_matched_anchor_count),
            "pde_query_matched_anchor_time_count": int(pde_query_matched_anchor_count),
            "pde_query_times_match_eval_grid": bool(pde_query_matches_eval_grid),
            "pde_query_times_match_user_query": bool(pde_query_matches_user_query),
            "inference_time_template": str(args.inference_time_template),
            "inference_time_seed": int(args.inference_time_seed),
            "inference_time_mask_mode": str(args.inference_time_mask_mode).lower(),
            "inference_obs_ratio": float(inference_obs_ratio),
            "inference_mask_seed": int(args.inference_mask_seed),
            "inference_time_kept_count_template": int(np.sum(inference_time_keep)),
            "inference_time_kept_count_mean_final": float(
                np.mean(np.sum(obs_time_mask, axis=1))
            ),
            "inference_time_total_count": int(nt),
            "temporal_inference_timing_scope": f"test split only; anchor_prepare = observed-time anchor core solve plus optional anchor decode; {temporal_head_scope_text}; decode_eval_metrics = eval-grid decode plus relL2 metric accumulation; pre_pde_total excludes PDE residual, SSIM, plots, and file saving",
            "temporal_anchor_prepare_seconds": float(
                inference_timings.get("anchor_prepare_seconds", float("nan"))
            ),
            "temporal_anchor_prepare_per_sequence_seconds": float(
                inference_timings.get(
                    "anchor_prepare_seconds_per_sequence", float("nan")
                )
            ),
            "temporal_head_forward_seconds": float(
                inference_timings.get("temporal_head_forward_seconds", float("nan"))
            ),
            "temporal_head_forward_per_sequence_seconds": float(
                inference_timings.get(
                    "temporal_head_forward_seconds_per_sequence", float("nan")
                )
            ),
            "temporal_decode_eval_metrics_seconds": float(
                inference_timings.get("decode_eval_metrics_seconds", float("nan"))
            ),
            "temporal_decode_eval_metrics_per_sequence_seconds": float(
                inference_timings.get(
                    "decode_eval_metrics_seconds_per_sequence", float("nan")
                )
            ),
            "temporal_inference_pre_pde_total_seconds": float(
                inference_timings.get("pre_pde_total_seconds", float("nan"))
            ),
            "temporal_inference_pre_pde_per_sequence_seconds": float(
                inference_timings.get(
                    "pre_pde_total_seconds_per_sequence", float("nan")
                )
            ),
            "observed_time_ratio_original": float(np.mean(obs_time_mask_original)),
            "observed_time_ratio": float(np.mean(obs_time_mask)),
            "pde_time_query_substeps": max(int(args.pde_time_query_substeps), 1),
            "pde_time_query_enabled": bool(pde_time_query_enabled),
            "pde_time_query_num_times": int(pde_time_values.size),
            "pde_time_query_source_dt": float(pde_source_time_dt),
            "pde_time_query_dt": float(pde_query_time_dt),
            "mse_full": mse_full,
            "rmse": rmse_full,
            "rmse_full": rmse_full,
            "mae_full": mae_full,
            "rel_l2_full": rel_l2_full,
            "rmse_missing": rmse_missing,
            "rel_l2_missing": rel_l2_missing,
            "pred_pde": float(test_pde_metrics.get("pred_pde", float("nan"))),
            "gt_pde": float(test_pde_metrics.get("gt_pde", float("nan"))),
            "mse_observed": mse_obs,
            "rmse_observed": rmse_obs,
            "mae_observed": mae_obs,
            "rel_l2_observed": rel_l2_obs,
            "mse_sensor": mse_obs,
            "rmse_sensor": rmse_obs,
            "mae_sensor": mae_obs,
            "rel_l2_sensor": rel_l2_obs,
            "mse_nonsensor": mse_nonsensor,
            "rmse_nonsensor": rmse_nonsensor,
            "mae_nonsensor": mae_nonsensor,
            "rel_l2_nonsensor": rel_l2_nonsensor,
            "gap_nonsensor_sensor": gap_nonsensor_sensor,
            "channel_metrics": channel_metrics,
            "mask_ratio_original": float(np.mean(mask_np)),
            "mask_ratio": float(np.mean(inference_mask_np)),
            "basis_path": basis_path,
            "core_path": core_path,
            "ftm_manifest_path": str(args.ftm_manifest_path),
            "config_path": str(args.source_config_path),
            "resolved_config_path": os.path.join(results_dir, "resolved_config.yaml"),
            "temporal_model_path": temporal_model_path,
            "data_manifest_path": data_manifest_path,
            "data_path": data_path,
            "metadata_path": metadata_path,
            "observation_data_path": data_path,
            "evaluation_ground_truth_data_path": evaluation_ground_truth_data_path,
            "evaluation_ground_truth_source": evaluation_ground_truth_source,
            "evaluation_protocol": "temporal_protocol",
            "protocol_rmse_mask_semantics": "sampled_at_another_time_per_trajectory",
        }
        metrics.update(test_pde_metrics)
        metrics.update(extra_metrics)
        metrics.update(temporal_protocol_metrics)
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(metrics), f, indent=2)
        print("temporal continuous reconstruction done")
        if int(args.skip_save_predictions) == 1:
            print(f"skipped predictions: {pred_save_path}")
        else:
            print(f"saved predictions: {pred_save_path}")
        print(f"saved metrics: {metrics_path}")
        print(f"saved field plots: {os.path.join(results_dir, 'plots', 'fields')}")
        print(
            f"full-field metrics: rmse={rmse_full:.6e} mae={mae_full:.6e} rel_l2={rel_l2_full:.6e}"
        )
        print(
            f"observed metrics: rmse={rmse_obs:.6e} mae={mae_obs:.6e} rel_l2={rel_l2_obs:.6e}"
        )
        print(
            f"sensor split metrics: sensor_relL2={rel_l2_obs:.6e} nonsensor_relL2={rel_l2_nonsensor:.6e} gap={gap_nonsensor_sensor:.6e}"
        )
        print(
            "missing-frame metrics:",
            f"rmse={rmse_missing:.6e}",
            f"rel_l2={rel_l2_missing:.6e}",
        )
        print(
            "temporal inference timing:",
            f"anchor={metrics['temporal_anchor_prepare_per_sequence_seconds']:.6f}s/seq",
            f"{'odernn'}={metrics['temporal_head_forward_per_sequence_seconds']:.6f}s/seq",
            f"decode={metrics['temporal_decode_eval_metrics_per_sequence_seconds']:.6f}s/seq",
            f"pre_pde_total={metrics['temporal_inference_pre_pde_per_sequence_seconds']:.6f}s/seq",
        )
        if extra_metrics:
            print()
        if test_pde_metrics:
            print(
                f"pde metrics: pred_pde={float(test_pde_metrics.get('pred_pde', float('nan'))):.6e} gt_pde={float(test_pde_metrics.get('gt_pde', float('nan'))):.6e} pred_query={str(test_pde_metrics.get('pred_pde_query_mode', 'unknown'))} gt_query={str(test_pde_metrics.get('gt_pde_query_mode', 'unknown'))}"
            )
    finally:
        if query_chunk_size_active:
            _restore_active_pde_query_chunk_size(
                basis_function, previous_query_chunk_size
            )
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def _interpolate_tensor_time_cubic(field_t, source_times_np, target_times_np):
    """Cubic Hermite interpolation along time, with linear fallback for short grids."""
    source = np.asarray(source_times_np, dtype=np.float64).reshape(-1)
    target = np.asarray(target_times_np, dtype=np.float64).reshape(-1)
    if _times_same_grid(source, target):
        return field_t
    if int(source.size) < 4:
        return _interpolate_tensor_time_linear(field_t, source, target)
    if int(field_t.shape[1]) != int(source.size):
        raise ValueError("Cubic time interpolation source length mismatch.")
    values = field_t.movedim(1, 0)
    slopes = torch.empty_like(values)
    dt = np.diff(source)
    slopes[0] = (values[1] - values[0]) / float(dt[0])
    slopes[-1] = (values[-1] - values[-2]) / float(dt[-1])
    for i in range(1, int(source.size) - 1):
        slopes[i] = (values[i + 1] - values[i - 1]) / float(
            source[i + 1] - source[i - 1]
        )
    outputs = []
    for x in target:
        i = int(
            np.clip(np.searchsorted(source, x, side="right") - 1, 0, source.size - 2)
        )
        h = float(source[i + 1] - source[i])
        s = float(np.clip((x - source[i]) / max(h, 1e-12), 0.0, 1.0))
        st = torch.as_tensor(s, dtype=field_t.dtype, device=field_t.device)
        ht = torch.as_tensor(h, dtype=field_t.dtype, device=field_t.device)
        h00 = 2 * st**3 - 3 * st**2 + 1
        h10 = st**3 - 2 * st**2 + st
        h01 = -2 * st**3 + 3 * st**2
        h11 = st**3 - st**2
        outputs.append(
            h00 * values[i]
            + h10 * ht * slopes[i]
            + h01 * values[i + 1]
            + h11 * ht * slopes[i + 1]
        )
    return torch.stack(outputs, dim=0).movedim(0, 1)


def _compute_temporal_pde_losses_from_field(
    *,
    pred_field,
    coord_uniques,
    dataset_type,
    pde_kwargs,
    channel_order,
    point_sample_ratio,
    point_max_points,
    sample_coords=None,
):
    dataset_type = str(dataset_type).lower()
    if dataset_type == "heat3d":
        coord_u, coord_v, coord_w = coord_uniques[:3]
        order = ("u", "v", "w")
        query_fn = build_dense_field_query_fn_3d(
            pred_field, coord_u, coord_v, coord_w, spatial_axis_order=order
        )
        return compute_heat3d_pde_loss_fd_query(
            query_fn=query_fn,
            sample_coords=(
                sample_coords
                if sample_coords is not None
                else _build_plot_sample_coords(
                    (coord_u, coord_v, coord_w),
                    point_sample_ratio,
                    point_max_points,
                    pred_field.device,
                    "temporal_gt_heat",
                    include_w=True,
                )
            ),
            coord_u=coord_u,
            coord_v=coord_v,
            coord_w=coord_w,
            time_coords_physical=pde_kwargs["time_coords_physical"],
            Lx=pde_kwargs["Lx"],
            Ly=pde_kwargs["Ly"],
            Lz=pde_kwargs["Lz"],
            alpha=pde_kwargs["alpha"],
            fd_dx=pde_kwargs.get("fd_dx"),
            fd_dy=pde_kwargs.get("fd_dy"),
            fd_dz=pde_kwargs.get("fd_dz"),
            fd_space_order=pde_kwargs.get("fd_space_order", 2),
        )
    periodic_u, periodic_v = _temporal_pde_axis_periodicity(dataset_type)
    query_fn = build_dense_field_query_fn_2d(
        pred_field,
        coord_uniques[0],
        coord_uniques[1],
        periodic_u=periodic_u,
        periodic_v=periodic_v,
    )
    return _compute_temporal_pde_losses_from_query_fn(
        query_fn=query_fn,
        coord_uniques=coord_uniques,
        dataset_type=dataset_type,
        pde_kwargs=pde_kwargs,
        channel_order=channel_order,
        point_sample_ratio=point_sample_ratio,
        point_max_points=point_max_points,
        sample_coords=sample_coords,
        pred_field=pred_field,
    )


def _compute_temporal_pde_losses_from_query_fn(
    *,
    query_fn,
    coord_uniques,
    dataset_type,
    pde_kwargs,
    channel_order,
    point_sample_ratio,
    point_max_points,
    sample_coords=None,
    pred_field=None,
):
    if sample_coords is None:
        coord_device = torch.as_tensor(coord_uniques[0]).device
        sample_coords = _sample_continuous_spatial_points(
            coord_uniques,
            sample_ratio=point_sample_ratio,
            max_points=point_max_points,
            device=coord_device,
        )
    if sample_coords is None:
        return {}
    dataset_type = str(dataset_type).lower()
    if dataset_type == "heat3d":
        if pred_field is None:
            raise ValueError(
                f"{dataset_type} PDE evaluation requires a dense field for GT queries."
            )
        return _compute_temporal_pde_losses_from_field(
            pred_field=pred_field,
            coord_uniques=coord_uniques,
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
            channel_order=channel_order,
            point_sample_ratio=point_sample_ratio,
            point_max_points=point_max_points,
            sample_coords=sample_coords,
        )
    if dataset_type == "ac":
        return compute_ac_pde_loss_fd_query(
            query_fn=query_fn,
            sample_coords=sample_coords,
            coord_u=coord_uniques[0],
            coord_v=coord_uniques[1],
            dt=pde_kwargs["dt"],
            Lx=pde_kwargs["Lx"],
            Ly=pde_kwargs["Ly"],
            ac_lambda=pde_kwargs["ac_lambda"],
            epsilon=pde_kwargs["epsilon"],
            fd_dx=pde_kwargs.get("fd_dx", None),
            fd_dy=pde_kwargs.get("fd_dy", None),
            fd_space_order=pde_kwargs.get("fd_space_order", 2),
            fd_time_order=pde_kwargs.get("fd_time_order", 2),
        )
    if dataset_type == "ns":
        if channel_order is None:
            raise ValueError("NS PDE evaluation requires the (u,v,omega) channels.")
        return compute_ns_pde_loss_fd_query(
            query_fn=query_fn,
            sample_coords=sample_coords,
            coord_u=coord_uniques[0],
            coord_v=coord_uniques[1],
            dt=pde_kwargs["dt"],
            Lx=pde_kwargs["Lx"],
            Ly=pde_kwargs["Ly"],
            nu=pde_kwargs["nu"],
            forcing_scale=pde_kwargs["forcing_scale"],
            channel_order=channel_order,
            fd_dx=pde_kwargs.get("fd_dx", None),
            fd_dy=pde_kwargs.get("fd_dy", None),
            fd_space_order=pde_kwargs.get("fd_space_order", 2),
            fd_time_order=pde_kwargs.get("fd_time_order", 2),
        )
    raise ValueError(
        f"Unsupported temporal PDE dataset_type={dataset_type!r}; expected AC, NS, heat3d, or ocean."
    )


def _can_use_ftm_continuous_pde_query_3d(
    core_like, basis_function, shared_basis_multichannel=False
):
    if (
        core_like is None
        or basis_function is None
        or (not hasattr(basis_function, "mode_w"))
    ):
        return False
    ndim = getattr(core_like, "ndim", None)
    if ndim is None:
        ndim = np.asarray(core_like).ndim
    ndim = int(ndim)
    return ndim == (6 if shared_basis_multichannel else 5)


def _build_ftm_continuous_query_fn_3d(
    *,
    core_t,
    basis_function,
    norm_stats,
    spatial_axis_order=("u", "v", "w"),
    shared_basis_multichannel=False,
):
    if not _can_use_ftm_continuous_pde_query_3d(
        core_t, basis_function, shared_basis_multichannel
    ):
        raise ValueError(
            "Direct FTM continuous PDE query requires a 3D per-time core and mode_w basis."
        )
    core_t = torch.as_tensor(
        core_t, dtype=torch.float32, device=next(basis_function.parameters()).device
    )

    def query_fn(query_coords):
        sampled_norm = _sample_field_with_mode_basis_3d(
            basis_function,
            core_t,
            query_coords,
            spatial_axis_order=spatial_axis_order,
            shared_basis_multichannel=bool(shared_basis_multichannel),
        )
        return inverse_normalize_tensor(sampled_norm, norm_stats)

    return query_fn


def _compute_temporal_pde_batched(
    *,
    pred_np,
    data_t,
    norm_stats,
    time_mask,
    pred_time_values_np,
    gt_time_values_np,
    target_time_values_np,
    batch_size,
    device,
    coord_uniques,
    dataset_type,
    pde_kwargs,
    channel_order,
    point_sample_ratio,
    point_max_points,
    pred_core_np=None,
    basis_function=None,
    shared_basis_multichannel=False,
):
    dataset_type = str(dataset_type).lower()
    pred_time_values = np.asarray(pred_time_values_np, dtype=np.float32).reshape(-1)
    gt_time_values = np.asarray(gt_time_values_np, dtype=np.float32).reshape(-1)
    target_time_values = np.asarray(target_time_values_np, dtype=np.float32).reshape(-1)
    time_mask = np.asarray(time_mask, dtype=bool).reshape(-1)
    if int(time_mask.size) != int(target_time_values.size):
        raise ValueError(
            f"time_mask length {int(time_mask.size)} != target_time_values length {int(target_time_values.size)}"
        )
    if not np.all(time_mask):
        target_time_values = target_time_values[time_mask]
    if int(target_time_values.size) < 3:
        return {}
    dense_query_mode = (
        "dense_grid_periodic_fd"
        if dataset_type == "ns" and channel_order is None
        else "dense_bilinear"
    )
    pred_pde_query_mode = (
        "ftm_continuous_basis"
        if _can_use_ftm_continuous_pde_query(
            pred_core_np,
            basis_function,
            shared_basis_multichannel=shared_basis_multichannel,
        )
        and (not (dataset_type == "ns" and channel_order is None))
        else dense_query_mode
    )
    if dataset_type == "heat3d":
        dense_query_mode = "dense_trilinear"
        pred_pde_query_mode = (
            "ftm_continuous_basis"
            if _can_use_ftm_continuous_pde_query_3d(
                pred_core_np, basis_function, shared_basis_multichannel
            )
            else dense_query_mode
        )
    pred_terms_sum = {}
    gt_terms_sum = {}
    total_weight = 0.0
    num_samples = int(pred_np.shape[0])
    batch_stride = 1
    for start in range(0, num_samples, batch_stride):
        end = min(start + batch_stride, num_samples)
        if dataset_type == "heat3d":
            sample_coords = _build_plot_sample_coords(
                coord_uniques[:3],
                sample_ratio=point_sample_ratio,
                max_points=point_max_points,
                device=device,
                seed_text=f"temporal_pde_eval|{dataset_type}|sample|{start}|{num_samples}",
                include_w=True,
            )
        else:
            sample_coords = _build_plot_sample_coords(
                coord_uniques,
                sample_ratio=point_sample_ratio,
                max_points=point_max_points,
                device=device,
                seed_text=f"temporal_pde_eval|{dataset_type}|sample|{start}|{num_samples}",
            )
        if sample_coords is None:
            continue
        gt_t = inverse_normalize_tensor(data_t[start:end].to(device), norm_stats)
        if dataset_type == "heat3d":
            pred_t = torch.as_tensor(
                pred_np[start:end], dtype=torch.float32, device=device
            )
            if pred_t.dim() == gt_t.dim() - 1 and int(gt_t.shape[-1]) == 1:
                pred_t = pred_t.unsqueeze(-1)
            elif gt_t.dim() == pred_t.dim() - 1 and int(pred_t.shape[-1]) == 1:
                gt_t = gt_t.unsqueeze(-1)
            if not _times_same_grid(pred_time_values, target_time_values):
                pred_t = _interpolate_tensor_time_cubic(
                    pred_t, pred_time_values, target_time_values
                )
            if not _times_same_grid(gt_time_values, target_time_values):
                gt_t = _interpolate_tensor_time_cubic(
                    gt_t, gt_time_values, target_time_values
                )
            if int(pred_t.shape[1]) < 3 or int(gt_t.shape[1]) < 3:
                continue
            eval_kwargs = dict(pde_kwargs)
            if dataset_type == "heat3d":
                eval_kwargs["time_coords_physical"] = _map_time_coords_to_physical(
                    target_time_values,
                    gt_time_values,
                    pde_kwargs.get("time_coords_physical", gt_time_values),
                )
            if (
                pred_core_np is not None
                and basis_function is not None
                and _can_use_ftm_continuous_pde_query_3d(
                    pred_core_np[start:end],
                    basis_function,
                    shared_basis_multichannel=shared_basis_multichannel,
                )
            ):
                cu, cv, cw, order = (
                    coord_uniques[0],
                    coord_uniques[1],
                    coord_uniques[2],
                    ("u", "v", "w"),
                )
                pred_query = _build_ftm_continuous_query_fn_3d(
                    core_t=pred_core_np[start:end],
                    basis_function=basis_function,
                    norm_stats=norm_stats,
                    spatial_axis_order=order,
                    shared_basis_multichannel=shared_basis_multichannel,
                )
                pred_losses = compute_heat3d_pde_loss_fd_query(
                    query_fn=pred_query,
                    sample_coords=sample_coords,
                    coord_u=cu,
                    coord_v=cv,
                    coord_w=cw,
                    time_coords_physical=eval_kwargs["time_coords_physical"],
                    Lx=eval_kwargs["Lx"],
                    Ly=eval_kwargs["Ly"],
                    Lz=eval_kwargs["Lz"],
                    alpha=eval_kwargs["alpha"],
                    fd_dx=eval_kwargs.get("fd_dx"),
                    fd_dy=eval_kwargs.get("fd_dy"),
                    fd_dz=eval_kwargs.get("fd_dz"),
                    fd_space_order=eval_kwargs.get("fd_space_order", 2),
                )
            else:
                pred_losses = _compute_temporal_pde_losses_from_field(
                    pred_field=pred_t,
                    coord_uniques=coord_uniques,
                    dataset_type=dataset_type,
                    pde_kwargs=eval_kwargs,
                    channel_order=channel_order,
                    point_sample_ratio=point_sample_ratio,
                    point_max_points=point_max_points,
                    sample_coords=sample_coords,
                )
            gt_losses = _compute_temporal_pde_losses_from_field(
                pred_field=gt_t,
                coord_uniques=coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=eval_kwargs,
                channel_order=channel_order,
                point_sample_ratio=point_sample_ratio,
                point_max_points=point_max_points,
                sample_coords=sample_coords,
            )
            pred_losses = ns_equation_export_terms(pred_losses, dataset_type)
            gt_losses = ns_equation_export_terms(gt_losses, dataset_type)
            if "pde" not in pred_losses or "pde" not in gt_losses:
                continue
            weight = float(end - start)
            for name, value in pred_losses.items():
                if torch.is_tensor(value) and value.dim() == 0:
                    pred_terms_sum[name] = pred_terms_sum.get(
                        name, 0.0
                    ) + weight * float(value.item())
            for name, value in gt_losses.items():
                if torch.is_tensor(value) and value.dim() == 0:
                    gt_terms_sum[name] = gt_terms_sum.get(name, 0.0) + weight * float(
                        value.item()
                    )
            total_weight += weight
            continue
        if dataset_type == "ns" and channel_order is None:
            pred_t = torch.as_tensor(
                pred_np[start:end], dtype=torch.float32, device=device
            )
            if pred_t.dim() == gt_t.dim() - 1 and int(gt_t.shape[-1]) == 1:
                pred_t = pred_t.unsqueeze(-1)
            elif gt_t.dim() == pred_t.dim() - 1 and int(pred_t.shape[-1]) == 1:
                gt_t = gt_t.unsqueeze(-1)
            if not _times_same_grid(pred_time_values, target_time_values):
                pred_t = _interpolate_tensor_time_linear(
                    pred_t, pred_time_values, target_time_values
                )
            elif np.any(time_mask):
                pred_t = pred_t[:, time_mask]
            if not _times_same_grid(gt_time_values, target_time_values):
                gt_t = _interpolate_tensor_time_linear(
                    gt_t, gt_time_values, target_time_values
                )
            elif np.any(time_mask):
                gt_t = gt_t[:, time_mask]
            if int(pred_t.shape[1]) < 3 or int(gt_t.shape[1]) < 3:
                continue
            pred_losses = _compute_temporal_pde_losses_from_field(
                pred_field=pred_t,
                coord_uniques=coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=channel_order,
                point_sample_ratio=point_sample_ratio,
                point_max_points=point_max_points,
                sample_coords=sample_coords,
            )
            gt_losses = _compute_temporal_pde_losses_from_field(
                pred_field=gt_t,
                coord_uniques=coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=channel_order,
                point_sample_ratio=point_sample_ratio,
                point_max_points=point_max_points,
                sample_coords=sample_coords,
            )
        else:
            use_direct_pred_query = pred_pde_query_mode == "ftm_continuous_basis"
            periodic_u, periodic_v = _temporal_pde_axis_periodicity(dataset_type)
            if use_direct_pred_query:
                pred_core_t = torch.as_tensor(
                    pred_core_np[start:end], dtype=torch.float32, device=device
                )
                if int(pred_core_t.shape[1]) != int(target_time_values.size):
                    raise ValueError(
                        f"Pred core time length {int(pred_core_t.shape[1])} != target_time_values length {int(target_time_values.size)}"
                    )
                if int(pred_core_t.shape[1]) < 3:
                    continue
                pred_query_fn = _build_ftm_continuous_query_fn_2d(
                    core_t=pred_core_t,
                    basis_function=basis_function,
                    norm_stats=norm_stats,
                    coord_uniques=coord_uniques,
                    shared_basis_multichannel=shared_basis_multichannel,
                )
                pred_field_for_loss = None
            else:
                pred_t = torch.as_tensor(
                    pred_np[start:end], dtype=torch.float32, device=device
                )
                if pred_t.dim() == gt_t.dim() - 1 and int(gt_t.shape[-1]) == 1:
                    pred_t = pred_t.unsqueeze(-1)
                elif gt_t.dim() == pred_t.dim() - 1 and int(pred_t.shape[-1]) == 1:
                    gt_t = gt_t.unsqueeze(-1)
                pred_query_fn = _build_spatiotemporal_dense_query_fn_2d(
                    pred_t,
                    pred_time_values,
                    target_time_values,
                    coord_uniques[0],
                    coord_uniques[1],
                    periodic_u=periodic_u,
                    periodic_v=periodic_v,
                )
                pred_field_for_loss = (
                    pred_t
                    if _times_same_grid(pred_time_values, target_time_values)
                    else None
                )
            gt_query_fn = _build_spatiotemporal_dense_query_fn_2d(
                gt_t,
                gt_time_values,
                target_time_values,
                coord_uniques[0],
                coord_uniques[1],
                periodic_u=periodic_u,
                periodic_v=periodic_v,
            )
            gt_field_for_loss = (
                gt_t if _times_same_grid(gt_time_values, target_time_values) else None
            )
            pred_losses = _compute_temporal_pde_losses_from_query_fn(
                query_fn=pred_query_fn,
                coord_uniques=coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=channel_order,
                point_sample_ratio=point_sample_ratio,
                point_max_points=point_max_points,
                sample_coords=sample_coords,
                pred_field=pred_field_for_loss,
            )
            gt_losses = _compute_temporal_pde_losses_from_query_fn(
                query_fn=gt_query_fn,
                coord_uniques=coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=channel_order,
                point_sample_ratio=point_sample_ratio,
                point_max_points=point_max_points,
                sample_coords=sample_coords,
                pred_field=gt_field_for_loss,
            )
        pred_losses = ns_equation_export_terms(pred_losses, dataset_type)
        gt_losses = ns_equation_export_terms(gt_losses, dataset_type)
        if "pde" not in pred_losses or "pde" not in gt_losses:
            continue
        weight = float(end - start)
        for name, value in pred_losses.items():
            if torch.is_tensor(value) and value.dim() == 0:
                pred_terms_sum[name] = pred_terms_sum.get(name, 0.0) + weight * float(
                    value.item()
                )
        for name, value in gt_losses.items():
            if torch.is_tensor(value) and value.dim() == 0:
                gt_terms_sum[name] = gt_terms_sum.get(name, 0.0) + weight * float(
                    value.item()
                )
        total_weight += weight
    if total_weight <= 0.0:
        return {}
    out = {}
    for name, value in pred_terms_sum.items():
        out[f"pred_{name}"] = float(value / total_weight)
    for name, value in gt_terms_sum.items():
        out[f"gt_{name}"] = float(value / total_weight)
    out["pred_pde_query_mode"] = pred_pde_query_mode
    out["gt_pde_query_mode"] = dense_query_mode
    return out


if __name__ == "__main__":
    main()
