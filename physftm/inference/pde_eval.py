import os
import torch
from physftm.artifacts.loading import inverse_normalize_tensor
from physftm.inference.core import _sampling_coord_uniques_for_inference
from physftm.pde.query_fd import (
    _inverse_norm_affine_channel_last,
    _sample_field_with_mode_basis_2d,
    build_dense_field_query_fn_2d,
    compute_ac_pde_loss_fd_query,
    compute_ns_pde_loss_fd_query,
    build_dense_field_query_fn_3d,
    compute_heat3d_pde_loss_fd_query,
    _sample_field_with_mode_basis_3d,
)
from physftm.pde.training import (
    _build_plot_sample_coords,
    _compute_query_sampled_pde_loss,
    _sample_continuous_time_points,
)


def ns_equation_export_terms(terms, dataset_type):
    """Export equation MSE without altering the training constraint sum."""
    if str(dataset_type).lower() == "ns" and "term_vorticity_residual" in terms:
        terms = dict(terms)
        terms["pde_constraint_sum"] = terms["pde"]
        terms["pde"] = terms["term_vorticity_residual"]
    return terms


class _InferenceBasisRef:

    def __init__(self, basis_function):
        self.basis_function = basis_function


def _infer_dataset_key(data_manifest_path=None, data_manifest=None, ftm_manifest=None):
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
    merged = " | ".join(text_parts).upper().replace("-", "_")
    for part in text_parts:
        token = os.path.basename(str(part).strip()).upper().replace("-", "_")
        if (
            token.startswith("NS")
            or "NS2D" in token
            or "UVOMEGA" in token
            or ("OMEGAONLY" in token)
        ):
            return "NS"
        if token.startswith("WAVE") or "WAVE_CASE" in token:
            return "WAVE"
        if "HEAT3D" in token or "HEAT_3D" in token:
            return "HEAT3D"
        if token.startswith("OCEAN") or "COPERNICUS" in token:
            return "OCEAN"
    if "NS2D" in merged or "UVOMEGA" in merged or "OMEGAONLY" in merged:
        return "NS"
    if "WAVE" in merged or "WAVE_CASE" in merged:
        return "WAVE"
    if "HEAT3D" in merged or "HEAT_3D" in merged:
        return "HEAT3D"
    if "OCEAN" in merged or "COPERNICUS" in merged:
        return "OCEAN"
    if "AC" in merged:
        return "AC"
    return None


def _resolve_pde_dataset_type(
    requested_type, *, data_manifest_path=None, data_manifest=None, ftm_manifest=None
):
    requested = str(requested_type or "auto").strip().lower()
    if requested == "none":
        return None
    if requested in ("ac", "ns", "heat3d", "ocean"):
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported pde_dataset_type={requested_type!r}")
    inferred = _infer_dataset_key(
        data_manifest_path=data_manifest_path,
        data_manifest=data_manifest,
        ftm_manifest=ftm_manifest,
    )
    if inferred is None:
        raise ValueError(
            f"Cannot infer pde_dataset_type from manifest_path={data_manifest_path!r}, data_name={(None if not isinstance(data_manifest, dict) else data_manifest.get('data_name'))!r}, ftm_data_name={(None if ftm_manifest is None else ftm_manifest.get('data_name'))!r}. Set --pde_dataset_type explicitly to ac, ns, heat3d, or none."
        )
    return str(inferred).lower()


def _build_pde_kwargs(args, dataset_type, pde_context=None):
    dataset_type = str(dataset_type).lower()
    if dataset_type == "ac":
        return dict(
            Lx=float(args.pde_Lx),
            Ly=float(args.pde_Ly),
            dt=0.01 if args.pde_dt is None else float(args.pde_dt),
            query_dt=0.001 if args.pde_query_dt is None else float(args.pde_query_dt),
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
            query_dt=0.01 if args.pde_query_dt is None else float(args.pde_query_dt),
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
            Ly=1.0 if float(args.pde_Ly) == 1.0 else float(args.pde_Ly),
            Lz=(
                1.0
                if float(getattr(args, "pde_Lz", 1.0)) == 1.0
                else float(getattr(args, "pde_Lz", 1.0))
            ),
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
    raise ValueError(
        f"Unsupported pde dataset_type={dataset_type!r}; expected AC, NS, WAVE, HEAT3D, or OCEAN."
    )


def _pde_axis_periodicity(dataset_type):
    dataset_type = str(dataset_type).lower()
    if dataset_type in ("ac", "ns"):
        return (True, True)
    if dataset_type == "heat3d":
        return (False, False)
    raise ValueError(
        f"Unsupported pde dataset_type={dataset_type!r}; expected AC, NS, WAVE, HEAT3D, or OCEAN."
    )


def _compute_pde_losses_from_field(
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
        order = ("u", "v", "w")
        coord_u, coord_v, coord_w = coord_uniques[:3]
        query_fn = build_dense_field_query_fn_3d(
            pred_field,
            coord_u,
            coord_v,
            coord_w,
            spatial_axis_order=order,
            periodic_u=False,
            periodic_v=False,
            periodic_w=False,
        )
        return compute_heat3d_pde_loss_fd_query(
            query_fn=query_fn,
            sample_coords=(
                sample_coords
                if sample_coords is not None
                else _build_plot_sample_coords(
                    (coord_u, coord_v, coord_w),
                    sample_ratio=point_sample_ratio,
                    max_points=point_max_points,
                    device=pred_field.device,
                    seed_text="gt_heat3d_pde",
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
    if sample_coords is None:
        sample_coords = _build_plot_sample_coords(
            coord_uniques,
            sample_ratio=point_sample_ratio,
            max_points=point_max_points,
            device=pred_field.device,
            seed_text=f"basis_recon_pde|{dataset_type}|{tuple(pred_field.shape)}",
        )
    if sample_coords is None:
        return {}
    periodic_u, periodic_v = _pde_axis_periodicity(dataset_type)
    query_fn = build_dense_field_query_fn_2d(
        pred_field,
        coord_uniques[0],
        coord_uniques[1],
        periodic_u=periodic_u,
        periodic_v=periodic_v,
    )
    dataset_type = str(dataset_type).lower()
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
            raise ValueError(
                "NS PDE evaluation now requires multichannel query order ('u', 'v', 'omega'); omega-only dense fallback has been removed."
            )
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
        f"Unsupported pde dataset_type={dataset_type!r}; expected AC, NS, or WAVE."
    )


def _build_ftm_native_query_fn(
    *,
    basis_function,
    core_batch,
    norm_stats,
    coord_uniques=None,
    shared_basis_multichannel,
):
    num_channels = int(core_batch.shape[2]) if bool(shared_basis_multichannel) else 1
    affine_scale, affine_bias = _inverse_norm_affine_channel_last(
        norm_stats,
        num_channels=num_channels,
        field_ndim=3,
        device=core_batch.device,
        dtype=core_batch.dtype,
    )

    def query_fn(query_coords):
        sampled = _sample_field_with_mode_basis_2d(
            basis_function,
            core_batch,
            query_coords,
            coord_uniques=coord_uniques,
            shared_basis_multichannel=shared_basis_multichannel,
        )
        return sampled * affine_scale + affine_bias

    return query_fn


def _compute_ftm_native_query_pde_batched(
    *,
    core_np,
    data_t,
    norm_stats,
    device,
    basis_function,
    shared_basis_multichannel,
    coord_uniques,
    dataset_type,
    pde_kwargs,
    channel_order,
    point_sample_ratio,
    point_max_points,
    core_parameterization="per_time",
    pde_context=None,
):
    dataset_type = str(dataset_type).lower()
    if dataset_type == "ns" and channel_order is None:
        raise ValueError(
            "NS PDE evaluation now requires multichannel query order ('u', 'v', 'omega'); omega-only dense fallback has been removed."
        )
    pred_terms_sum = {}
    gt_terms_sum = {}
    total_weight = 0.0
    num_samples = int(core_np.shape[0])
    core_parameterization = str(core_parameterization).lower()
    sampling_coord_uniques = _sampling_coord_uniques_for_inference(
        coord_uniques, core_parameterization
    )
    if dataset_type == "heat3d":
        logical_u = torch.as_tensor(
            coord_uniques[0], dtype=torch.float32, device=device
        )
        logical_v = torch.as_tensor(
            coord_uniques[1], dtype=torch.float32, device=device
        )
        logical_w = torch.as_tensor(
            coord_uniques[2], dtype=torch.float32, device=device
        )
        spatial_axis_order = ("u", "v", "w")
    else:
        logical_u = torch.as_tensor(
            sampling_coord_uniques[0], dtype=torch.float32, device=device
        )
        logical_v = torch.as_tensor(
            sampling_coord_uniques[1], dtype=torch.float32, device=device
        )
        logical_w = None
        spatial_axis_order = ("u", "v")
    for sample_idx in range(num_samples):
        core_t = torch.as_tensor(
            core_np[sample_idx : sample_idx + 1], dtype=torch.float32, device=device
        )
        gt_t = inverse_normalize_tensor(
            torch.as_tensor(
                data_t[sample_idx : sample_idx + 1], dtype=torch.float32, device=device
            ),
            norm_stats,
        )
        if gt_t.dim() == 4:
            gt_t = gt_t.unsqueeze(-1)
        if int(gt_t.shape[1]) < 3:
            continue
        if dataset_type == "heat3d":
            sample_coords = _build_plot_sample_coords(
                (logical_u, logical_v, logical_w),
                sample_ratio=point_sample_ratio,
                max_points=point_max_points,
                device=device,
                seed_text=f"basis_recon_pde|{dataset_type}|sample|{sample_idx}|{num_samples}",
                include_w=True,
            )
        else:
            sample_coords = _build_plot_sample_coords(
                (logical_u, logical_v),
                sample_ratio=point_sample_ratio,
                max_points=point_max_points,
                device=device,
                seed_text=f"basis_recon_pde|{dataset_type}|sample|{sample_idx}|{num_samples}",
            )
        if sample_coords is None:
            continue
        sample_pde_kwargs = pde_kwargs
        sample_coord_uniques = coord_uniques
        sample_sampling_coord_uniques = (
            coord_uniques
            if dataset_type in ("heat3d", "ocean")
            else sampling_coord_uniques
        )
        sample_time_coords = _sample_continuous_time_points(
            sample_coord_uniques,
            sample_ratio=point_sample_ratio,
            max_points=point_max_points,
            device=device,
            core_parameterization=core_parameterization,
        )
        with torch.no_grad():
            if dataset_type != "heat3d":
                pred_query_fn = _build_ftm_native_query_fn(
                    basis_function=basis_function,
                    core_batch=core_t,
                    norm_stats=norm_stats,
                    coord_uniques=sample_coord_uniques,
                    shared_basis_multichannel=shared_basis_multichannel,
                )
                pred_losses = _compute_query_sampled_pde_loss(
                    query_fn=pred_query_fn,
                    sample_coords=sample_coords,
                    sample_time_coords=sample_time_coords,
                    coord_uniques=sample_coord_uniques,
                    core_parameterization=core_parameterization,
                    dataset_type=dataset_type,
                    pde_kwargs=sample_pde_kwargs,
                    channel_order=channel_order,
                )
            if dataset_type == "heat3d":
                affine_scale, affine_bias = _inverse_norm_affine_channel_last(
                    norm_stats,
                    num_channels=(
                        int(core_t.shape[2]) if shared_basis_multichannel else 1
                    ),
                    field_ndim=3,
                    device=device,
                    dtype=core_t.dtype,
                )

                def query3d(query_coords):
                    return (
                        _sample_field_with_mode_basis_3d(
                            basis_function,
                            core_t,
                            query_coords,
                            spatial_axis_order=spatial_axis_order,
                            shared_basis_multichannel=shared_basis_multichannel,
                        )
                        * affine_scale
                        + affine_bias
                    )

                pred_losses = compute_heat3d_pde_loss_fd_query(
                    query_fn=query3d,
                    sample_coords=sample_coords,
                    coord_u=logical_u,
                    coord_v=logical_v,
                    coord_w=logical_w,
                    time_coords_physical=sample_pde_kwargs["time_coords_physical"],
                    Lx=sample_pde_kwargs["Lx"],
                    Ly=sample_pde_kwargs["Ly"],
                    Lz=sample_pde_kwargs["Lz"],
                    alpha=sample_pde_kwargs["alpha"],
                    fd_dx=sample_pde_kwargs.get("fd_dx"),
                    fd_dy=sample_pde_kwargs.get("fd_dy"),
                    fd_dz=sample_pde_kwargs.get("fd_dz"),
                    fd_space_order=sample_pde_kwargs.get("fd_space_order", 2),
                )
            gt_losses = _compute_pde_losses_from_field(
                pred_field=gt_t,
                coord_uniques=sample_sampling_coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=sample_pde_kwargs,
                channel_order=channel_order,
                point_sample_ratio=point_sample_ratio,
                point_max_points=point_max_points,
                sample_coords=sample_coords,
            )
        pred_losses = ns_equation_export_terms(pred_losses, dataset_type)
        gt_losses = ns_equation_export_terms(gt_losses, dataset_type)
        if "pde" not in pred_losses or "pde" not in gt_losses:
            continue
        total_weight += 1.0
        for name, value in pred_losses.items():
            if torch.is_tensor(value) and value.dim() == 0:
                pred_terms_sum[name] = pred_terms_sum.get(name, 0.0) + float(
                    value.item()
                )
        for name, value in gt_losses.items():
            if torch.is_tensor(value) and value.dim() == 0:
                gt_terms_sum[name] = gt_terms_sum.get(name, 0.0) + float(value.item())
    if total_weight <= 0.0:
        return {}
    out = {}
    for name, value in pred_terms_sum.items():
        out[f"pred_{name}"] = float(value / total_weight)
    for name, value in gt_terms_sum.items():
        out[f"gt_{name}"] = float(value / total_weight)
    return out


def _compute_pde_batched(
    *,
    pred_np,
    data_t,
    norm_stats,
    batch_size,
    device,
    coord_uniques,
    dataset_type,
    pde_kwargs,
    channel_order,
    point_sample_ratio,
    point_max_points,
    pde_context=None,
):
    dataset_type = str(dataset_type).lower()
    if dataset_type == "heat3d":
        pred_terms_sum, gt_terms_sum, total_weight = ({}, {}, 0.0)
        for start in range(0, int(pred_np.shape[0]), max(int(batch_size), 1)):
            end = min(start + max(int(batch_size), 1), int(pred_np.shape[0]))
            pred_t = torch.as_tensor(
                pred_np[start:end], dtype=torch.float32, device=device
            )
            gt_t = inverse_normalize_tensor(data_t[start:end].to(device), norm_stats)
            if pred_t.dim() == gt_t.dim() - 1:
                pred_t = pred_t.unsqueeze(-1)
            if gt_t.dim() == pred_t.dim() - 1:
                gt_t = gt_t.unsqueeze(-1)
            sample_coords = _build_plot_sample_coords(
                coord_uniques[:3],
                sample_ratio=point_sample_ratio,
                max_points=point_max_points,
                device=device,
                seed_text=f"cubic_heat3d|{start}",
                include_w=True,
            )
            pred_losses = _compute_pde_losses_from_field(
                pred_field=pred_t,
                coord_uniques=coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=channel_order,
                point_sample_ratio=point_sample_ratio,
                point_max_points=point_max_points,
                sample_coords=sample_coords,
            )
            gt_losses = _compute_pde_losses_from_field(
                pred_field=gt_t,
                coord_uniques=coord_uniques,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=channel_order,
                point_sample_ratio=point_sample_ratio,
                point_max_points=point_max_points,
                sample_coords=sample_coords,
            )
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
        if total_weight <= 0.0:
            return {}
        return {
            **{f"pred_{n}": v / total_weight for n, v in pred_terms_sum.items()},
            **{f"gt_{n}": v / total_weight for n, v in gt_terms_sum.items()},
        }
    pred_terms_sum = {}
    gt_terms_sum = {}
    total_weight = 0.0
    num_samples = int(pred_np.shape[0])
    for start in range(0, num_samples, int(batch_size)):
        end = min(start + int(batch_size), num_samples)
        sample_coords = _build_plot_sample_coords(
            coord_uniques,
            sample_ratio=point_sample_ratio,
            max_points=point_max_points,
            device=device,
            seed_text=f"basis_recon_pde|{dataset_type}|{start}|{end}|{num_samples}",
        )
        if sample_coords is None:
            continue
        pred_t = torch.as_tensor(pred_np[start:end], dtype=torch.float32, device=device)
        gt_t = inverse_normalize_tensor(data_t[start:end].to(device), norm_stats)
        if pred_t.dim() == gt_t.dim() - 1 and int(gt_t.shape[-1]) == 1:
            pred_t = pred_t.unsqueeze(-1)
        elif gt_t.dim() == pred_t.dim() - 1 and int(pred_t.shape[-1]) == 1:
            gt_t = gt_t.unsqueeze(-1)
        if int(pred_t.shape[1]) < 3:
            continue
        pred_losses = _compute_pde_losses_from_field(
            pred_field=pred_t,
            coord_uniques=coord_uniques,
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
            channel_order=channel_order,
            point_sample_ratio=point_sample_ratio,
            point_max_points=point_max_points,
            sample_coords=sample_coords,
        )
        gt_losses = _compute_pde_losses_from_field(
            pred_field=gt_t,
            coord_uniques=coord_uniques,
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
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
    return out
