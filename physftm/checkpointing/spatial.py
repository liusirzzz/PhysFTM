"""RKHS FTM checkpoint artifact helpers."""

import json
import os
import time
import numpy as np
import scipy.io as scio
import torch
import torch.distributed as dist
from physftm.data.axes import spatial_axis_order_for_dataset

RKHS_ARTIFACT_VERSION = 3


def build_spatial_method_tag(config):
    parts = [str("matern32").lower()]
    if str(getattr(config, "core_update_mode", "cg")).lower() == "adam":
        parts.append("coreadam")
    if str(config.basis_update_mode).lower() == "solve":
        parts.append("solve")
    return "_".join(parts)


def build_checkpoint_bundle_paths(save_dir, data_name, rank):
    rank_tag = f"{rank[0]}x{rank[1]}x{rank[2]}"
    core_path = os.path.join(save_dir, f"core_{data_name}_{rank_tag}.mat")
    basis_path = os.path.join(save_dir, f"basis_{data_name}_{rank_tag}.pth")
    manifest_path = os.path.join(save_dir, f"ftm_{data_name}_{rank_tag}.json")
    return (core_path, basis_path, manifest_path)


def _jsonify(value):
    if torch.is_tensor(value):
        return _jsonify(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _coord_list(coord):
    return (
        torch.as_tensor(coord, dtype=torch.float32).detach().cpu().reshape(-1).tolist()
    )


def _build_coordinates_manifest(config, ind_input, t_coords):
    if len(ind_input) == 3:
        axis_order = spatial_axis_order_for_dataset(
            getattr(config, "dataset_type", ""), 3
        )
        coords_by_axis = dict(zip(axis_order, ind_input))
        return {
            "order": ["t", *axis_order],
            "source": "training_unique_coordinates",
            "t": _coord_list(t_coords),
            "u": _coord_list(coords_by_axis["u"]),
            "v": _coord_list(coords_by_axis["v"]),
            "w": _coord_list(coords_by_axis["w"]),
        }
    u_ind, v_ind = ind_input
    return {
        "order": ["t", "u", "v"],
        "source": "training_unique_coordinates",
        "t": _coord_list(t_coords),
        "u": _coord_list(u_ind),
        "v": _coord_list(v_ind),
    }


def _build_inducing_coordinates_manifest(
    basis_function, core_parameterization, spatial_axis_order=None
):
    factors = [basis_function.mode_u, basis_function.mode_v]
    if hasattr(basis_function, "mode_w"):
        factors.append(basis_function.mode_w)
    factor_order = tuple(
        spatial_axis_order or (("u", "v", "w") if len(factors) == 3 else ("u", "v"))
    )
    out = {"selection": "uniform_training_grid"}
    out.update(
        {
            axis: _coord_list(factor.kernel.z)
            for axis, factor in zip(factor_order, factors)
        }
    )
    out["order"] = list(factor_order)
    return out


def _kernel_lengthscale_manifest(
    basis_function, core_parameterization, spatial_axis_order=None
):
    factors = [basis_function.mode_u, basis_function.mode_v]
    if hasattr(basis_function, "mode_w"):
        factors.append(basis_function.mode_w)
    factor_order = tuple(
        spatial_axis_order or (("u", "v", "w") if len(factors) == 3 else ("u", "v"))
    )
    out = {
        axis: float(factor.kernel.log_lengthscale.detach().exp().cpu())
        for axis, factor in zip(factor_order, factors)
    }
    return out


def gather_core_table_for_saving(
    model, train_sampler, ddp_state, device, attr_name="tucker_core"
):
    """Gather per-rank updated core rows to rank0 and merge into a full core table."""
    if not ddp_state["distributed"]:
        core = getattr(model, attr_name).detach().cpu()
        return core
    local_indices = torch.as_tensor(
        list(iter(train_sampler)), dtype=torch.long, device=device
    )
    core_table = getattr(model.module, attr_name)
    local_core = (
        core_table.detach()
        .index_select(0, local_indices.to(device=core_table.device))
        .to(device=device, non_blocking=True)
        .contiguous()
    )
    world_size = ddp_state["world_size"]
    local_n = torch.tensor([local_indices.numel()], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)
    pad_n = max_n - local_indices.numel()
    if pad_n > 0:
        local_indices_pad = torch.cat(
            [local_indices, torch.full((pad_n,), -1, device=device, dtype=torch.long)],
            dim=0,
        )
        pad_shape = (pad_n,) + tuple(local_core.shape[1:])
        local_core_pad = torch.cat(
            [local_core, torch.zeros(pad_shape, device=device, dtype=local_core.dtype)],
            dim=0,
        )
    else:
        local_indices_pad = local_indices
        local_core_pad = local_core
    gathered_idx = [torch.empty_like(local_indices_pad) for _ in range(world_size)]
    gathered_core = [torch.empty_like(local_core_pad) for _ in range(world_size)]
    dist.all_gather(gathered_idx, local_indices_pad)
    dist.all_gather(gathered_core, local_core_pad)
    if ddp_state["rank"] != 0:
        return None
    full_core = getattr(model.module, attr_name).detach().cpu().clone()
    for r in range(world_size):
        n_r = sizes[r]
        idx_r = gathered_idx[r][:n_r].cpu()
        core_r = gathered_core[r][:n_r].cpu()
        valid = idx_r >= 0
        if valid.any():
            full_core[idx_r[valid]] = core_r[valid]
    return full_core.cpu()


def save_checkpoint_bundle(
    *,
    save_dir,
    experiment_root_dir,
    save_kind,
    core_to_save,
    basis_to_save,
    norm_stats,
    ind_input,
    t_coords,
    config,
    rank,
    shared_basis_multichannel,
    num_channels,
    best_rmse,
    best_mae,
    best_rel_l2,
    eval_rmse,
    eval_mae,
    eval_rel_l2,
    best_step,
    step,
    train_start_time,
):
    os.makedirs(save_dir, exist_ok=True)
    core_save_path, basis_save_path, manifest_save_path = build_checkpoint_bundle_paths(
        save_dir, config.data_name, rank
    )
    model_type = "rkhs_ftm"
    rank_axis_order = "per_time"
    basis_class = (
        "RKHSFunctionalTucker3D"
        if hasattr(basis_to_save, "mode_w")
        else "RKHSFunctionalTucker2D"
    )
    scio.savemat(core_save_path, {"core": core_to_save.numpy()})
    torch.save(
        {
            "format_version": RKHS_ARTIFACT_VERSION,
            "artifact_type": "rkhs_basis_state",
            "state_dict": basis_to_save.state_dict(),
        },
        basis_save_path,
    )
    is_3d = hasattr(basis_to_save, "mode_w") and "per_time" == "per_time"
    spatial_axis_order = spatial_axis_order_for_dataset(
        getattr(config, "dataset_type", ""), 3 if is_3d else 2
    )
    coordinates = _build_coordinates_manifest(
        config, ind_input=ind_input, t_coords=t_coords
    )
    inducing_coordinates = _build_inducing_coordinates_manifest(
        basis_to_save, "per_time", spatial_axis_order=spatial_axis_order
    )
    normalization = _jsonify(
        {
            "norm_mode": norm_stats["norm_mode"],
            "data_min": norm_stats["data_min"],
            "data_max": norm_stats["data_max"],
            "data_mean": norm_stats["data_mean"],
            "data_std": norm_stats["data_std"],
            "map_nonnegative": norm_stats.get("map_nonnegative", False),
            "nonnegative_shift": norm_stats.get("nonnegative_shift", 0.0),
            "map_signed": norm_stats.get("map_signed", False),
            "signed_center": norm_stats.get("signed_center", 0.0),
            "signed_half_range": norm_stats.get("signed_half_range", 1.0),
        }
    )
    training_config = {
        "core_update_mode": str(config.core_update_mode),
        "core_learning_rate": float(config.core_learning_rate),
        "lambda_core_ridge": float(config.lambda_core_ridge),
        "beta_core_tv": float(config.beta_core_tv or 0.0),
        "als_core_cg_iter": int(config.als_core_cg_iter),
        "als_core_cg_tol": float(config.als_core_cg_tol),
        "als_core_steps": int(config.als_core_steps),
        "basis_update_mode": str(config.basis_update_mode),
        "basis_solve_value_ridge": float(config.basis_solve_value_ridge),
        "basis_solve_target_clip": float(config.basis_solve_target_clip),
        "learning_rate": float(config.learning_rate),
        "lambda_pde_point": float(config.lambda_pde_point),
        "pde_start_step": int(config.pde_start_step),
        "pde_point_sample_ratio": float(config.pde_point_sample_ratio),
        "pde_point_max_points": int(config.pde_point_max_points),
        "pde_query_chunk_size": int(config.pde_query_chunk_size),
        "pde_point_sample_mode": str(
            getattr(config, "pde_point_sample_mode", "continuous")
        ),
        "pde_monitor_always": bool(config.pde_monitor_always),
        "pde_monitor_point": bool(config.pde_monitor_point),
        "source_config_path": getattr(config, "source_config_path", None),
        "seed": int(getattr(config, "seed", 123)),
    }
    configured_lengthscales = {
        "u": None if config.lengthscale_u is None else float(config.lengthscale_u),
        "v": None if config.lengthscale_v is None else float(config.lengthscale_v),
    }
    if hasattr(basis_to_save, "mode_w"):
        configured_lengthscales["w"] = (
            None
            if getattr(config, "lengthscale_w", None) is None
            else float(config.lengthscale_w)
        )
    basis_config = {
        "class": basis_class,
        "parameterization": "per_time",
        "rank": list(rank),
        "learnable_kernel": bool(False),
        "kernel_type": str("matern32"),
        "configured_lengthscales": configured_lengthscales,
        "effective_lengthscales": _kernel_lengthscale_manifest(
            basis_to_save, "per_time", spatial_axis_order=spatial_axis_order
        ),
        "spatial_axis_order": list(spatial_axis_order),
        "num_inducing": {
            key: len(value)
            for key, value in inducing_coordinates.items()
            if key in ("t", "u", "v", "w")
        },
        "coef_scale": float(config.coef_scale),
    }
    if hasattr(basis_to_save, "mode_w"):
        basis_config["configured_lengthscales"]["w"] = (
            None
            if getattr(config, "lengthscale_w", None) is None
            else float(config.lengthscale_w)
        )
    ftm_manifest = {
        "artifact_version": RKHS_ARTIFACT_VERSION,
        "data_name": config.data_name,
        "rank": list(rank),
        "core_parameterization": "per_time",
        "rank_axis_order": rank_axis_order,
        "spatial_axis_order": list(spatial_axis_order),
        "basis_class": basis_class,
        "core_path": core_save_path,
        "basis_path": basis_save_path,
        "source_data_manifest_path": config.data_manifest_path,
        "experiment_root_dir": str(experiment_root_dir),
        "norm_mode": norm_stats["norm_mode"],
        "save_kind": str(save_kind),
        "model_type": model_type,
        "pde_start_step": int(config.pde_start_step),
        "pde_monitor_always": bool(config.pde_monitor_always),
        "pde_monitor_point": bool(config.pde_monitor_point),
        "core_update_mode": str(config.core_update_mode),
        "core_learning_rate": float(config.core_learning_rate),
        "core_table_on_cpu": str(config.core_update_mode).lower() == "adam",
        "lambda_core_ridge": float(config.lambda_core_ridge),
        "beta_core_tv": float(config.beta_core_tv or 0.0),
        "als_core_cg_iter": int(config.als_core_cg_iter),
        "als_core_cg_tol": float(config.als_core_cg_tol),
        "als_core_steps": int(config.als_core_steps),
        "basis_update_mode": str(config.basis_update_mode),
        "shared_basis_multichannel": bool(shared_basis_multichannel),
        "num_channels": int(num_channels),
        "basis_solve_value_ridge": float(config.basis_solve_value_ridge),
        "basis_solve_target_clip": float(config.basis_solve_target_clip),
        "kernel_type": str("matern32"),
        "paths": {
            "core_path": os.path.basename(core_save_path),
            "basis_path": os.path.basename(basis_save_path),
        },
        "model": {
            "type": model_type,
            "core_parameterization": "per_time",
            "rank_axis_order": rank_axis_order,
            "rank": list(rank),
            "shared_basis_multichannel": bool(shared_basis_multichannel),
            "num_channels": int(num_channels),
        },
        "basis": basis_config,
        "core": {"format": "mat", "variables": ["core"], "layout": "per_time"},
        "coordinates": coordinates,
        "inducing_coordinates": inducing_coordinates,
        "normalization": normalization,
        "decode": {
            "field_layout": "NTDHW[C]" if hasattr(basis_to_save, "mode_w") else "NTHWC",
            "core_parameterization": "per_time",
            "shared_basis_multichannel": bool(shared_basis_multichannel),
            "coordinate_order": ["t", *spatial_axis_order],
            "normalization": "apply inverse normalization from manifest.normalization",
        },
        "training": training_config,
        "provenance": {
            "source_data_manifest_path": config.data_manifest_path,
            "source_config_path": getattr(config, "source_config_path", None),
            "config_snapshot_path": os.path.join(
                str(experiment_root_dir), "config.yaml"
            ),
            "resolved_config_path": os.path.join(
                str(experiment_root_dir), "resolved_config.json"
            ),
            "experiment_root_dir": str(experiment_root_dir),
            "save_kind": str(save_kind),
            "schema_note": "artifact v3 separates full decode coordinates from RKHS inducing coordinates; basis.pth stores a state_dict and core.mat stores only core.",
        },
        "metrics": {
            "best_val_rmse": float(best_rmse),
            "best_val_mae": float(best_mae),
            "best_val_rel_l2": float(best_rel_l2),
            "best_step": int(best_step),
            "eval_rmse": float(eval_rmse),
            "eval_mae": float(eval_mae),
            "eval_rel_l2": float(eval_rel_l2),
            "eval_metric_space": "physical",
            "eval_metric_split": "train_full_field",
            "eval_step": int(step),
        },
        "best_val_rmse": float(best_rmse),
        "best_val_mae": float(best_mae),
        "best_val_rel_l2": float(best_rel_l2),
        "best_step": int(best_step),
        "eval_rmse": float(eval_rmse),
        "eval_mae": float(eval_mae),
        "eval_rel_l2": float(eval_rel_l2),
        "eval_metric_space": "physical",
        "eval_metric_split": "train_full_field",
        "eval_step": int(step),
        "training_duration_seconds": round(time.time() - train_start_time, 1),
    }
    with open(manifest_save_path, "w", encoding="utf-8") as f:
        json.dump(ftm_manifest, f, indent=2)
    return (core_save_path, basis_save_path, manifest_save_path)
