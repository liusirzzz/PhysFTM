"""Snapshot diagnostics for PhysFTM training."""

import os
import numpy as np
import torch
from physftm.artifacts.loading import inverse_normalize_tensor
from physftm.decoding.tucker import decode_spatial_from_precomputed_basis
from physftm.pde.training import (
    _build_plot_sample_coords,
    _heat3d_nonuniform_time_diff,
    _compute_offgrid_point_residual_terms,
    _compute_per_time_ac_factor_pde_terms_chunk,
    _inverse_norm_affine_channel_last,
    _normalize_pde_point_sample_mode,
    _sample_mask_conditioned_spatial_coords_for_sample,
    _sample_continuous_time_points,
    _sample_field_with_mode_basis_2d,
    _spatial_coord_uniques,
)
from physftm.training.runtime import current_model
from physftm.training.wrapper import broadcast_mask_channels as _broadcast_mask_channels


def _snapshot_spatial_basis(basis_module, ind_input, core_parameterization):
    """Evaluate the basis in the layout expected by the shared decoder."""
    basis_module.mode = "training"
    if getattr(basis_module, "spatial_dim", 2) == 3:
        U, V, W = basis_module(input_ind_train=ind_input)
        return (U, V, (W,))
    U, V = basis_module(input_ind_train=ind_input)
    return (U, V, None)


@torch.no_grad()
def _compute_per_time_ac_factor_snapshot_terms(
    *, basis_function, core_batch, sample_coords, coord_uniques, norm_stats, pde_kwargs
):
    affine_scale, affine_bias = _inverse_norm_affine_channel_last(
        norm_stats,
        num_channels=1,
        field_ndim=3,
        device=core_batch.device,
        dtype=core_batch.dtype,
    )
    total_points = int(sample_coords.shape[0])
    configured_chunk_size = int(pde_kwargs.get("query_chunk_size", 0) or 0)
    chunk_size = (
        total_points
        if configured_chunk_size <= 0
        else min(configured_chunk_size, total_points)
    )
    term_chunks = {}
    scalar_terms = {}
    for start in range(0, total_points, chunk_size):
        end = min(start + chunk_size, total_points)
        chunk_terms = _compute_per_time_ac_factor_pde_terms_chunk(
            basis_function=basis_function,
            core_batch=core_batch,
            sample_coords=sample_coords[start:end],
            coord_uniques=coord_uniques,
            affine_scale=affine_scale,
            affine_bias=affine_bias,
            pde_kwargs=pde_kwargs,
        )
        for name, value in chunk_terms.items():
            value_cpu = value.detach().float().cpu()
            if name.startswith("fd_"):
                if name not in scalar_terms:
                    scalar_terms[name] = value_cpu
            else:
                term_chunks.setdefault(name, []).append(value_cpu)
        del chunk_terms
    merged = {name: torch.cat(chunks, dim=-1) for name, chunks in term_chunks.items()}
    merged.update(scalar_terms)
    return merged


def _pde_snapshot_channel_names(channel_order, num_channels):
    if channel_order is None:
        names = (
            ["field"]
            if int(num_channels) == 1
            else [f"ch{idx}" for idx in range(int(num_channels))]
        )
    else:
        names = [str(x) for x in channel_order]
    if len(names) < int(num_channels):
        names = names + [f"ch{idx}" for idx in range(len(names), int(num_channels))]
    return names[: int(num_channels)]


def parse_pde_snapshot_sample_indices(config):
    indices = getattr(config, "plot_pde_residual_sample_indices", None)
    if indices:
        if not isinstance(indices, (list, tuple)):
            raise ValueError(
                f"plot_pde_residual_sample_indices must be a YAML list, got {indices!r}."
            )
        return [int(item) for item in indices]
    return [int(config.plot_pde_residual_sample_index)]


def save_reconstruction_snapshot(
    *,
    model,
    ddp_state,
    batch_idx,
    data_batch,
    mask_batch,
    ind_input,
    norm_stats,
    shared_basis_multichannel,
    experiment_root_dir,
    step_tag,
    sample_in_batch=0,
):
    """Write reconstruction PNGs using the pre-refactor directory layout."""
    model_ref = current_model(model, ddp_state)
    basis_module = model_ref.basis_function
    model_device = model_ref.compute_device()
    batch_idx = batch_idx.to(model_device, non_blocking=True)
    data_batch = data_batch.to(model_device, non_blocking=True)
    mask_batch = mask_batch.to(model_device, non_blocking=True)
    core_batch = model_ref.get_core_batch(batch_idx, detach=True, device=model_device)
    with torch.no_grad():
        U, V, basis_extra = _snapshot_spatial_basis(basis_module, ind_input, "per_time")
        total_out = decode_spatial_from_precomputed_basis(
            U,
            V,
            basis_extra,
            core_batch,
            shared_basis_multichannel=shared_basis_multichannel,
            core_parameterization="per_time",
        )
        pred_phys = (
            inverse_normalize_tensor(total_out, norm_stats).detach().float().cpu()
        )
        gt_phys = (
            inverse_normalize_tensor(data_batch, norm_stats).detach().float().cpu()
        )
        mask_cpu = mask_batch.detach().float().cpu()
        mask_cpu = _broadcast_mask_channels(mask_cpu, pred_phys)
    if pred_phys.dim() == 4:
        pred_phys = pred_phys.unsqueeze(-1)
        gt_phys = gt_phys.unsqueeze(-1)
        mask_cpu = mask_cpu.unsqueeze(-1) if mask_cpu.dim() == 4 else mask_cpu
    num_channels = int(pred_phys.shape[-1])
    channel_names = _pde_snapshot_channel_names(None, num_channels)
    out_dir = os.path.join(experiment_root_dir, "reconstruction_snapshots")
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(sample_in_batch, (list, tuple, set)):
        sample_indices = [int(x) for x in sample_in_batch]
    else:
        sample_indices = [int(sample_in_batch)]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for raw_sample_idx in sample_indices:
        sample_idx = max(0, min(int(raw_sample_idx), int(pred_phys.shape[0]) - 1))
        sample_label = (
            int(batch_idx[sample_idx].detach().cpu().item())
            if int(batch_idx.numel()) > sample_idx
            else sample_idx
        )
        nt = int(pred_phys.shape[1])
        time_ids = np.linspace(0, nt - 1, num=min(3, nt), dtype=int)
        if time_ids.size == 0:
            continue
        if time_ids.size < 3:
            time_ids = np.pad(time_ids, (0, 3 - time_ids.size), mode="edge")
        sample_dir = os.path.join(
            out_dir, step_tag, f"sample_batch{sample_idx:02d}_global{sample_label:05d}"
        )
        latest_sample_dir = os.path.join(
            out_dir, "latest", f"sample_batch{sample_idx:02d}_global{sample_label:05d}"
        )
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(latest_sample_dir, exist_ok=True)
        n_rows = len(time_ids) * num_channels
        fig, axes = plt.subplots(n_rows, 3, figsize=(13.5, 3.8 * n_rows), squeeze=False)
        row = 0
        for tid in time_ids.tolist():
            for channel_idx in range(num_channels):
                channel_name = channel_names[channel_idx]
                gt_img = gt_phys[sample_idx, tid, :, :, channel_idx].numpy()
                pred_img = pred_phys[sample_idx, tid, :, :, channel_idx].numpy()
                mask_img = mask_cpu[sample_idx, tid, :, :, channel_idx].numpy()
                input_img = gt_img.copy()
                input_img[mask_img <= 0] = np.nan
                field_abs = max(
                    (
                        float(np.nanmax(np.abs(input_img)))
                        if np.any(np.isfinite(input_img))
                        else 0.0
                    ),
                    float(np.abs(pred_img).max()),
                    float(np.abs(gt_img).max()),
                    1e-12,
                )
                panels = [
                    (input_img, f"Input observed t={tid} {channel_name}"),
                    (pred_img, f"Pred t={tid} {channel_name}"),
                    (gt_img, f"GT t={tid} {channel_name}"),
                ]
                for col, (img, title) in enumerate(panels):
                    ax = axes[row, col]
                    im = ax.imshow(
                        img,
                        cmap="RdBu_r",
                        vmin=-field_abs,
                        vmax=field_abs,
                        origin="lower",
                    )
                    ax.set_title(title)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                row += 1
        fig.suptitle(
            f"Reconstruction snapshot {step_tag} sample={sample_label}", fontsize=14
        )
        fig.tight_layout()
        save_name = f"reconstruction_{step_tag}_sample{sample_label:05d}.png"
        for target_dir in (sample_dir, latest_sample_dir):
            fig.savefig(
                os.path.join(target_dir, save_name), dpi=160, bbox_inches="tight"
            )
        plt.close(fig)


def save_pde_residual_snapshot(
    *,
    model,
    ddp_state,
    batch_idx,
    data_batch,
    mask_batch,
    ind_input,
    norm_stats,
    pde_kwargs,
    point_sample_ratio,
    point_max_points,
    shared_basis_multichannel,
    experiment_root_dir,
    step_tag,
    sample_in_batch=0,
    pde_context=None,
):
    """Write PDE residual PNGs while preserving existing smoke-visible paths."""
    dataset_type = str(pde_kwargs.get("dataset_type", "ac")).lower()
    point_sample_mode = _normalize_pde_point_sample_mode(
        pde_kwargs.get("point_sample_mode", "continuous")
    )
    model_ref = current_model(model, ddp_state)
    basis_module = model_ref.basis_function
    model_device = model_ref.compute_device()
    batch_idx = batch_idx.to(model_device, non_blocking=True)
    data_batch = data_batch.to(model_device, non_blocking=True)
    mask_batch = (
        None if mask_batch is None else mask_batch.to(model_device, non_blocking=True)
    )
    core_batch = model_ref.get_core_batch(batch_idx, detach=True, device=model_device)
    with torch.no_grad():
        U, V, basis_extra = _snapshot_spatial_basis(basis_module, ind_input, "per_time")
        total_out = decode_spatial_from_precomputed_basis(
            U,
            V,
            basis_extra,
            core_batch,
            shared_basis_multichannel=shared_basis_multichannel,
            core_parameterization="per_time",
        )
        pred_phys = inverse_normalize_tensor(total_out, norm_stats)
        gt_phys = inverse_normalize_tensor(data_batch, norm_stats)
    num_channels = int(pred_phys.shape[-1]) if pred_phys.dim() == 5 else 1
    channel_order = pde_kwargs.get("channel_order", None)
    channel_names = _pde_snapshot_channel_names(channel_order, num_channels)
    time_trim = int(pde_kwargs.get("fd_time_order", 2)) // 2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spatial_u, spatial_v = _spatial_coord_uniques(
        ind_input, core_parameterization="per_time"
    )
    x_min = float(torch.as_tensor(spatial_u).min())
    x_max = float(torch.as_tensor(spatial_u).max())
    y_min = float(torch.as_tensor(spatial_v).min())
    y_max = float(torch.as_tensor(spatial_v).max())
    out_dir = os.path.join(experiment_root_dir, "pde_residual_snapshots")
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(sample_in_batch, (list, tuple, set)):
        sample_indices = [int(x) for x in sample_in_batch]
    else:
        sample_indices = [int(sample_in_batch)]
    if dataset_type == "heat3d":
        field = pred_phys[..., 0]
        time_coords = torch.as_tensor(
            pde_kwargs["time_coords_physical"], dtype=field.dtype, device=field.device
        ).reshape(-1)
        u_t = _heat3d_nonuniform_time_diff(field, time_coords)
        dx = float(pde_kwargs.get("fd_dx") or 0.0) or float(pde_kwargs["Lx"]) / max(
            int(field.shape[2]) - 1, 1
        )
        dy = float(pde_kwargs.get("fd_dy") or 0.0) or float(pde_kwargs["Ly"]) / max(
            int(field.shape[3]) - 1, 1
        )
        dz = float(pde_kwargs.get("fd_dz") or 0.0) or float(pde_kwargs["Lz"]) / max(
            int(field.shape[4]) - 1, 1
        )
        center = field[:, 1:-1, 1:-1, 1:-1, 1:-1]
        lap = (
            (
                field[:, 1:-1, 2:, 1:-1, 1:-1]
                - 2.0 * center
                + field[:, 1:-1, :-2, 1:-1, 1:-1]
            )
            / dx**2
            + (
                field[:, 1:-1, 1:-1, 2:, 1:-1]
                - 2.0 * center
                + field[:, 1:-1, 1:-1, :-2, 1:-1]
            )
            / dy**2
            + (
                field[:, 1:-1, 1:-1, 1:-1, 2:]
                - 2.0 * center
                + field[:, 1:-1, 1:-1, 1:-1, :-2]
            )
            / dz**2
        )
        residual_sq = (
            u_t[:, :, 1:-1, 1:-1, 1:-1] - float(pde_kwargs["alpha"]) * lap
        ).square()
        coord_u, coord_v = (ind_input[0], ind_input[1])
        x_min, x_max = (
            float(torch.as_tensor(coord_u).min()),
            float(torch.as_tensor(coord_u).max()),
        )
        y_min, y_max = (
            float(torch.as_tensor(coord_v).min()),
            float(torch.as_tensor(coord_v).max()),
        )
        z_inner = int(residual_sq.shape[2] // 2)
        z_full = z_inner + 1
        for sample_order, raw_sample_idx in enumerate(sample_indices):
            sample_idx = max(0, min(int(raw_sample_idx), int(pred_phys.shape[0]) - 1))
            sample_label = (
                int(batch_idx[sample_idx].detach().cpu().item())
                if int(batch_idx.numel()) > sample_idx
                else sample_idx
            )
            num_times = int(residual_sq.shape[1])
            time_ids = np.linspace(0, num_times - 1, num=min(3, num_times), dtype=int)
            if time_ids.size == 0:
                continue
            if time_ids.size < 3:
                time_ids = np.pad(time_ids, (0, 3 - time_ids.size), mode="edge")
            sample_dir = os.path.join(
                out_dir,
                step_tag,
                f"sample_batch{sample_idx:02d}_global{sample_label:05d}",
            )
            latest_sample_dir = os.path.join(
                out_dir,
                "latest",
                f"sample_batch{sample_idx:02d}_global{sample_label:05d}",
            )
            os.makedirs(sample_dir, exist_ok=True)
            os.makedirs(latest_sample_dir, exist_ok=True)
            for tid in time_ids.tolist():
                center_tid = int(tid) + 1
                gt_img = (
                    gt_phys[sample_idx, center_tid, z_full, :, :, 0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                pred_img = (
                    pred_phys[sample_idx, center_tid, z_full, :, :, 0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                err_img = pred_img - gt_img
                pred_loss_img = (
                    residual_sq[sample_idx, tid, z_inner].detach().float().cpu().numpy()
                )
                field_abs = max(
                    float(np.abs(gt_img).max()), float(np.abs(pred_img).max()), 1e-12
                )
                err_abs = max(float(np.abs(err_img).max()), 1e-12)
                loss_max = max(float(pred_loss_img.max()), 1e-12)
                log_loss_img = np.log10(np.clip(pred_loss_img, a_min=1e-12, a_max=None))
                log_vmin = float(np.percentile(log_loss_img, 5.0))
                log_vmax = float(np.percentile(log_loss_img, 99.0))
                if log_vmax <= log_vmin:
                    log_vmax = log_vmin + 1e-06
                max_idx = np.unravel_index(
                    int(np.argmax(pred_loss_img)), pred_loss_img.shape
                )
                max_loss = float(pred_loss_img[max_idx])
                info = f"dataset=heat3d\nstep={step_tag}\nsample_batch={sample_idx}\nsample_global={sample_label}\ntime_idx={center_tid}\nz_idx={z_full}\nmax res^2={max_loss:.2e}\ndx={dx:.3g}, dy={dy:.3g}, dz={dz:.3g}"
                fig, axes = plt.subplots(2, 4, figsize=(18, 7.6), squeeze=False)
                panels = [
                    (gt_img, "GT center-z slice", -field_abs, field_abs, "RdBu_r"),
                    (pred_img, "Pred center-z slice", -field_abs, field_abs, "RdBu_r"),
                    (err_img, "Pred-GT", -err_abs, err_abs, "RdBu_r"),
                    (np.abs(err_img), "|Error|", 0.0, err_abs, "magma"),
                    (pred_loss_img, "PDE residual squared", 0.0, loss_max, "viridis"),
                    (log_loss_img, "PDE residual log10", log_vmin, log_vmax, "magma"),
                ]
                for col, (img, title, vmin, vmax, cmap) in enumerate(panels):
                    ax = axes[0 if col < 4 else 1, col if col < 4 else col - 4]
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
                axes[1, 2].axis("off")
                axes[1, 2].text(
                    0.02,
                    0.98,
                    info,
                    transform=axes[1, 2].transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                )
                axes[1, 3].axis("off")
                fig.suptitle(
                    f"PDE Residual Snapshot | heat3d | {step_tag} | sample={sample_label} | time={center_tid}",
                    fontsize=14,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                filename = f"heat3d_{step_tag}_sample{sample_label:05d}_time{center_tid:04d}.png"
                fig.savefig(
                    os.path.join(sample_dir, filename), dpi=180, bbox_inches="tight"
                )
                fig.savefig(
                    os.path.join(
                        latest_sample_dir,
                        f"heat3d_latest_sample{sample_label:05d}_time{center_tid:04d}.png",
                    ),
                    dpi=180,
                    bbox_inches="tight",
                )
                if sample_order == 0 and tid == time_ids.tolist()[0]:
                    fig.savefig(
                        os.path.join(out_dir, "latest.png"),
                        dpi=180,
                        bbox_inches="tight",
                    )
                plt.close(fig)
        return
    ns_query_multichannel = (
        dataset_type == "ns"
        and channel_order is not None
        and all(
            (
                name in tuple((str(x) for x in channel_order))
                for name in ("u", "v", "omega")
            )
        )
    )
    if dataset_type == "ns" and (not ns_query_multichannel):
        raise ValueError("NS snapshots require the (u,v,omega) channels.")
    for sample_order, raw_sample_idx in enumerate(sample_indices):
        sample_idx = max(0, min(int(raw_sample_idx), int(pred_phys.shape[0]) - 1))
        sample_label = (
            int(batch_idx[sample_idx].detach().cpu().item())
            if int(batch_idx.numel()) > sample_idx
            else sample_idx
        )
        if point_sample_mode == "continuous":
            sample_coords = _build_plot_sample_coords(
                ind_input,
                sample_ratio=point_sample_ratio,
                max_points=point_max_points,
                device=pred_phys.device,
                seed_text=f"{experiment_root_dir}|{step_tag}|sample{sample_idx}",
            )
        else:
            sample_coords = _sample_mask_conditioned_spatial_coords_for_sample(
                coord_uniques=ind_input,
                mask_sample=mask_batch[sample_idx],
                sample_ratio=point_sample_ratio,
                max_points=point_max_points,
                device=pred_phys.device,
                sample_mode=point_sample_mode,
                seed_text=f"{experiment_root_dir}|{step_tag}|sample{sample_idx}",
            )
        if sample_coords is None:
            continue
        pred_sample_coords = sample_coords.detach()
        plot_coords = pred_sample_coords.detach().float().cpu()
        plot_kwargs = pde_kwargs
        sample_t_coords = _sample_continuous_time_points(
            ind_input,
            sample_ratio=point_sample_ratio,
            max_points=0,
            device=pred_phys.device,
            core_parameterization="per_time",
        )
        plot_x = plot_coords[:, 0].numpy()
        plot_y = plot_coords[:, 1].numpy()
        pred_field_all = (
            pred_phys[sample_idx, time_trim:-time_trim].detach().float().cpu()
        )
        gt_field_all = gt_phys[sample_idx, time_trim:-time_trim].detach().float().cpu()
        if pred_field_all.dim() == 3:
            pred_field_all = pred_field_all.unsqueeze(-1)
            gt_field_all = gt_field_all.unsqueeze(-1)
        valid_num_times = int(pred_field_all.shape[0])
        if valid_num_times <= 0:
            continue

        def _pred_query_fn(query_coords):
            sampled = _sample_field_with_mode_basis_2d(
                basis_module,
                core_batch[sample_idx : sample_idx + 1],
                query_coords,
                coord_uniques=ind_input,
                shared_basis_multichannel=shared_basis_multichannel,
            )
            affine_scale, affine_bias = _inverse_norm_affine_channel_last(
                norm_stats,
                num_channels=num_channels,
                field_ndim=3,
                device=sampled.device,
                dtype=sampled.dtype,
            )
            return sampled * affine_scale + affine_bias

        if dataset_type == "ac" and num_channels == 1:
            pred_terms = _compute_per_time_ac_factor_snapshot_terms(
                basis_function=basis_module,
                core_batch=core_batch[sample_idx : sample_idx + 1],
                sample_coords=pred_sample_coords,
                coord_uniques=ind_input,
                norm_stats=norm_stats,
                pde_kwargs=plot_kwargs,
            )
        else:
            pred_terms = _compute_offgrid_point_residual_terms(
                query_fn=_pred_query_fn,
                sample_coords=pred_sample_coords,
                sample_t_coords=sample_t_coords,
                coord_uniques=ind_input,
                pde_kwargs={**plot_kwargs, "core_parameterization": "per_time"},
            )
        pred_point_loss = pred_terms["residual_sq"][0].detach().float().cpu()
        snapshot_term_names = [
            name
            for name in pred_terms.keys()
            if name
            not in ("residual_sq", "fd_dx", "fd_dy", "fd_dz", "fd_dphi", "fd_dtheta")
        ]
        snapshot_term_arrays = {
            name: value[0].detach().float().cpu()
            for name, value in pred_terms.items()
            if name in snapshot_term_names
            and torch.is_tensor(value)
            and (value.dim() >= 3)
        }
        num_times = int(pred_point_loss.shape[0])
        time_ids = np.linspace(0, num_times - 1, num=min(3, num_times), dtype=int)
        if time_ids.size == 0:
            continue
        if time_ids.size < 3:
            time_ids = np.pad(time_ids, (0, 3 - time_ids.size), mode="edge")
        sample_dir = os.path.join(
            out_dir, step_tag, f"sample_batch{sample_idx:02d}_global{sample_label:05d}"
        )
        latest_sample_dir = os.path.join(
            out_dir, "latest", f"sample_batch{sample_idx:02d}_global{sample_label:05d}"
        )
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(latest_sample_dir, exist_ok=True)
        for tid in time_ids.tolist():
            center_tid = tid + time_trim
            pred_loss_img = pred_point_loss[tid].numpy()
            loss_max = max(float(pred_loss_img.max()), 1e-12)
            log_loss_img = np.log10(np.clip(pred_loss_img, a_min=1e-12, a_max=None))
            log_vmin = float(np.percentile(log_loss_img, 5.0))
            log_vmax = float(np.percentile(log_loss_img, 99.0))
            if not np.isfinite(log_vmin):
                log_vmin = float(np.min(log_loss_img))
            if not np.isfinite(log_vmax):
                log_vmax = float(np.max(log_loss_img))
            if log_vmax <= log_vmin:
                log_vmax = log_vmin + 1e-06
            hotspot_thr = float(np.percentile(pred_loss_img, 99.0))
            hotspot_mask = pred_loss_img >= hotspot_thr
            max_idx = int(np.argmax(pred_loss_img))
            max_x = float(plot_x[max_idx])
            max_y = float(plot_y[max_idx])
            max_loss = float(pred_loss_img[max_idx])
            max_term_lines = [
                f"max res^2={max_loss:.2e}",
                f"x={max_x:.3g}, y={max_y:.3g}",
            ]
            for term_name in snapshot_term_names[:8]:
                term_arr = snapshot_term_arrays.get(term_name, None)
                if term_arr is None:
                    continue
                term_img = term_arr[tid].numpy()
                max_term_lines.append(f"{term_name}={float(term_img[max_idx]):.2e}")
            n_rows = num_channels + 1
            fig, axes = plt.subplots(
                n_rows, 4, figsize=(18, 3.8 * n_rows), squeeze=False
            )
            for channel_idx in range(num_channels):
                channel_name = channel_names[channel_idx]
                gt_img = gt_field_all[tid, :, :, channel_idx].numpy()
                pred_img = pred_field_all[tid, :, :, channel_idx].numpy()
                err_img = pred_img - gt_img
                field_abs = max(
                    float(np.abs(pred_img).max()), float(np.abs(gt_img).max()), 1e-12
                )
                err_abs = max(float(np.abs(err_img).max()), 1e-12)
                ax = axes[channel_idx, 0]
                im_gt = ax.imshow(
                    gt_img,
                    cmap="RdBu_r",
                    vmin=-field_abs,
                    vmax=field_abs,
                    origin="lower",
                )
                ax.set_title(f"GT {channel_name}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im_gt, ax=ax, fraction=0.046, pad=0.04)
                ax = axes[channel_idx, 1]
                im_pred = ax.imshow(
                    pred_img,
                    cmap="RdBu_r",
                    vmin=-field_abs,
                    vmax=field_abs,
                    origin="lower",
                )
                ax.set_title(f"Pred {channel_name}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im_pred, ax=ax, fraction=0.046, pad=0.04)
                ax = axes[channel_idx, 2]
                im_err = ax.imshow(
                    err_img, cmap="RdBu_r", vmin=-err_abs, vmax=err_abs, origin="lower"
                )
                ax.set_title(f"Pred-GT {channel_name}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)
                ax = axes[channel_idx, 3]
                im_abserr = ax.imshow(
                    np.abs(err_img),
                    cmap="magma",
                    vmin=0.0,
                    vmax=err_abs,
                    origin="lower",
                )
                ax.set_title(f"|Error| {channel_name}")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im_abserr, ax=ax, fraction=0.046, pad=0.04)
            residual_row = num_channels
            ax = axes[residual_row, 0]
            sc_pred = ax.scatter(
                plot_x,
                plot_y,
                c=pred_loss_img,
                s=6,
                cmap="viridis",
                vmin=0.0,
                vmax=loss_max,
                linewidths=0,
            )
            ax.set_title("PDE residual squared raw")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            fig.colorbar(sc_pred, ax=ax, fraction=0.046, pad=0.04)
            ax = axes[residual_row, 1]
            sc_hot = ax.scatter(
                plot_x,
                plot_y,
                c=log_loss_img,
                s=8,
                cmap="magma",
                vmin=log_vmin,
                vmax=log_vmax,
                linewidths=0,
            )
            if np.any(hotspot_mask):
                ax.scatter(
                    plot_x[hotspot_mask],
                    plot_y[hotspot_mask],
                    s=18,
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=0.7,
                )
            ax.scatter([max_x], [max_y], s=28, marker="x", c="lime", linewidths=1.0)
            ax.set_title("PDE residual log10 + top1%")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            fig.colorbar(sc_hot, ax=ax, fraction=0.046, pad=0.04)
            ax = axes[residual_row, 2]
            ax.axis("off")
            ax.text(
                0.02,
                0.98,
                "\n".join(max_term_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="0.6",
                ),
            )
            ax = axes[residual_row, 3]
            ax.axis("off")
            ax.text(
                0.02,
                0.98,
                f"dataset={dataset_type}\nstep={step_tag}\nsample_batch={sample_idx}\nsample_global={sample_label}\ntime_idx={center_tid}\nchannels={','.join(channel_names)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="0.6",
                ),
            )
            fig.suptitle(
                f"PDE Residual Snapshot | {dataset_type} | {step_tag} | sample={sample_label} | time={center_tid}",
                fontsize=14,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            filename = f"{dataset_type}_{step_tag}_sample{sample_label:05d}_time{center_tid:04d}.png"
            out_path = os.path.join(sample_dir, filename)
            latest_path = os.path.join(
                latest_sample_dir,
                f"{dataset_type}_latest_sample{sample_label:05d}_time{center_tid:04d}.png",
            )
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            fig.savefig(latest_path, dpi=200, bbox_inches="tight")
            if sample_order == 0 and tid == time_ids.tolist()[0]:
                fig.savefig(
                    os.path.join(out_dir, "latest.png"), dpi=200, bbox_inches="tight"
                )
            plt.close(fig)
