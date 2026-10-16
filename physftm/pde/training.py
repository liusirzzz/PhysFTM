"""Training-time PDE sampling helpers for PhysFTM.

This module is the shared home for code that was historically embedded in
`train_spatial.py`: channel conventions, continuous/query sampling, continuous query
contractions, and the training-time PDE loss dispatcher.
"""

import hashlib
import math
import numpy as np
import torch

try:
    from torch.utils.checkpoint import checkpoint as _checkpoint
except Exception:
    _checkpoint = None


def _build_plot_sample_coords(
    coord_uniques, sample_ratio, max_points, device, seed_text, include_w=False
):
    if bool(include_w):
        u_uni, v_uni = (coord_uniques[0], coord_uniques[1])
    else:
        u_uni, v_uni = _spatial_coord_uniques(
            coord_uniques, core_parameterization="per_time"
        )
    w_uni = coord_uniques[2] if bool(include_w) and len(coord_uniques) > 2 else None
    total = (
        int(torch.as_tensor(u_uni).numel())
        * int(torch.as_tensor(v_uni).numel())
        * (int(torch.as_tensor(w_uni).numel()) if w_uni is not None else 1)
    )
    sample_ratio = float(sample_ratio)
    max_points = int(max_points or 0)
    if max_points > 0:
        n_points = min(max_points, total)
    else:
        n_points = min(max(1, int(math.ceil(total * max(sample_ratio, 0.0)))), total)
    if n_points <= 0:
        return None
    seed = int(hashlib.sha1(str(seed_text).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    def _sample_axis(coord_uni, count):
        coord = (
            torch.as_tensor(coord_uni, dtype=torch.float32).reshape(-1).cpu().numpy()
        )
        if coord.size == 0:
            raise ValueError("Cannot sample a coordinate axis with no values.")
        if int(coord.shape[0]) >= 3:
            lo = float(coord[1])
            hi = float(coord[-2])
        else:
            lo = float(coord.min())
            hi = float(coord.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            midpoint = lo if np.isfinite(lo) else hi
            if not np.isfinite(midpoint):
                midpoint = 0.0
            return torch.full(
                (count,), float(midpoint), dtype=torch.float32, device=device
            )
        span = hi - lo
        eps = 1e-06 * span
        if span <= 2.0 * eps:
            vals = np.full((count,), 0.5 * (lo + hi), dtype=np.float32)
        else:
            vals = rng.uniform(lo + eps, hi - eps, size=count).astype(np.float32)
        return torch.from_numpy(vals).to(device=device)

    u_coords = _sample_axis(u_uni, n_points)
    v_coords = _sample_axis(v_uni, n_points)
    if w_uni is not None:
        return torch.stack([u_coords, v_coords, _sample_axis(w_uni, n_points)], dim=-1)
    return torch.stack([u_coords, v_coords], dim=-1)


def pde_channel_order(dataset_type, num_channels=1):
    dataset_type = str(dataset_type).lower()
    if dataset_type == "ns" and int(num_channels) == 3:
        return ("u", "v", "omega")
    if dataset_type == "ocean" and int(num_channels) == 2:
        return ("u", "v")
    return None


def _normalize_pde_point_sample_mode(value):
    text = str(value).strip().lower()
    if text not in ("continuous", "observed_grid", "unobserved_grid"):
        raise ValueError(
            "point_sample_mode must be continuous, observed_grid or unobserved_grid"
        )
    return text


def _broadcast_channel_last_stat(stat, num_channels, field_ndim, device, dtype):
    arr = torch.as_tensor(stat, dtype=dtype, device=device)
    if arr.ndim == 0:
        return arr.reshape((1,) * field_ndim)
    if arr.ndim == 1 and int(arr.numel()) == int(num_channels):
        return arr.reshape((1,) * (field_ndim - 1) + (int(num_channels),))
    while arr.ndim < field_ndim:
        arr = arr.unsqueeze(0)
    return arr


def _inverse_norm_affine_channel_last(
    norm_stats, num_channels, field_ndim, device, dtype
):
    scale = torch.ones((1,) * field_ndim, device=device, dtype=dtype)
    bias = torch.zeros((1,) * field_ndim, device=device, dtype=dtype)
    norm_mode = str(norm_stats.get("norm_mode", "none"))
    if norm_mode in ("zscore", "channelwise_zscore"):
        std = torch.clamp(
            _broadcast_channel_last_stat(
                norm_stats.get("data_std", 1.0),
                num_channels=num_channels,
                field_ndim=field_ndim,
                device=device,
                dtype=dtype,
            ),
            min=1e-08,
        )
        mean = _broadcast_channel_last_stat(
            norm_stats.get("data_mean", 0.0),
            num_channels=num_channels,
            field_ndim=field_ndim,
            device=device,
            dtype=dtype,
        )
        scale = scale * std
        bias = bias * std + mean
    elif norm_mode not in ("none", ""):
        raise ValueError(f"Unsupported norm_mode for sampled PDE loss: {norm_mode}")
    return (scale, bias)


def _core_parameterization_from_basis(basis_function) -> str:
    return "per_time"


def _pde_time_step(pde_kwargs, core_parameterization="per_time") -> float:
    return float(pde_kwargs["dt"])
    query_dt = pde_kwargs.get("query_dt", None)
    return float(pde_kwargs["dt"] if query_dt is None else query_dt)


def _set_active_pde_query_chunk_size(basis_function, chunk_size):
    prev = getattr(basis_function, "_pde_query_chunk_size", None)
    basis_function._pde_query_chunk_size = int(chunk_size)
    return prev


def _restore_active_pde_query_chunk_size(basis_function, prev):
    if prev is None:
        if hasattr(basis_function, "_pde_query_chunk_size"):
            delattr(basis_function, "_pde_query_chunk_size")
    else:
        basis_function._pde_query_chunk_size = int(prev)


def _spatial_coord_uniques(coord_uniques, core_parameterization="per_time"):
    return (coord_uniques[0], coord_uniques[1])


def _num_channels_from_core(
    core_batch, shared_basis_multichannel=False, core_parameterization="per_time"
):
    if not shared_basis_multichannel:
        return 1
    return int(core_batch.shape[2])


def _sample_continuous_spatial_points(
    coord_uniques,
    sample_ratio,
    max_points,
    device,
    core_parameterization="per_time",
    allow_full_sample=False,
    spatial_axis_order=None,
):
    if spatial_axis_order is None:
        spatial_axis_order = ("u", "v", "w") if len(coord_uniques) > 2 else ("u", "v")
    semantic = {"u": None, "v": None, "w": None}
    for axis, value in zip(tuple(spatial_axis_order), tuple(coord_uniques)):
        semantic[str(axis)] = value
    u_uni, v_uni = (semantic["u"], semantic["v"])
    w_uni = semantic["w"]
    total = (
        int(torch.as_tensor(u_uni).numel())
        * int(torch.as_tensor(v_uni).numel())
        * (int(torch.as_tensor(w_uni).numel()) if w_uni is not None else 1)
    )
    sample_ratio = float(sample_ratio)
    max_points = int(max_points or 0)
    if max_points <= 0 and sample_ratio >= 1.0 and (not bool(allow_full_sample)):
        return None
    if max_points > 0:
        n_points = min(max_points, total)
    else:
        n_points = min(max(1, int(math.ceil(total * max(sample_ratio, 0.0)))), total)
    if n_points <= 0:
        return None

    def _rand_axis(coord_uni, count):
        coord = torch.as_tensor(coord_uni, dtype=torch.float32, device=device).reshape(
            -1
        )
        if int(coord.numel()) >= 3:
            lo = coord[1]
            hi = coord[-2]
        else:
            lo = coord.min()
            hi = coord.max()
        span = (hi - lo).clamp_min(1e-08)
        eps = 1e-06 * span
        if bool((span <= 2.0 * eps).item()):
            return torch.full(
                (count,), float(0.5 * (lo + hi)), device=device, dtype=coord.dtype
            )
        return (
            lo
            + eps
            + torch.rand(count, device=device, dtype=coord.dtype) * (span - 2.0 * eps)
        )

    u_coords = _rand_axis(u_uni, n_points)
    v_coords = _rand_axis(v_uni, n_points)
    if w_uni is not None:
        return torch.stack([u_coords, v_coords, _rand_axis(w_uni, n_points)], dim=-1)
    return torch.stack([u_coords, v_coords], dim=-1)


def _heat3d_nonuniform_time_diff(field, time_coords):
    t = torch.as_tensor(time_coords, dtype=field.dtype, device=field.device).reshape(-1)
    if int(field.shape[1]) != int(t.numel()) or int(t.numel()) < 3:
        raise ValueError(
            "Heat3D requires at least 3 frames and matching physical time coordinates."
        )
    hp = t[1:-1] - t[:-2]
    hf = t[2:] - t[1:-1]
    cp = -hf / (hp * (hp + hf))
    cc = (hf - hp) / (hp * hf)
    cf = hp / (hf * (hp + hf))
    shape = (1, -1) + (1,) * (field.dim() - 2)
    return (
        cp.reshape(shape) * field[:, :-2]
        + cc.reshape(shape) * field[:, 1:-1]
        + cf.reshape(shape) * field[:, 2:]
    )


def _sample_continuous_time_points(
    coord_uniques, sample_ratio, max_points, device, core_parameterization="per_time"
):
    return None
    t_uni = torch.as_tensor(
        coord_uniques[0], dtype=torch.float32, device=device
    ).reshape(-1)
    total = int(t_uni.numel())
    sample_ratio = float(sample_ratio)
    max_points = int(max_points or 0)
    if max_points <= 0 and sample_ratio >= 1.0:
        return None
    if max_points > 0:
        n_points = min(max_points, total)
    else:
        n_points = min(max(1, int(math.ceil(total * max(sample_ratio, 0.0)))), total)
    if n_points <= 0:
        return None
    if int(t_uni.numel()) >= 3:
        lo = t_uni[1]
        hi = t_uni[-2]
    else:
        lo = t_uni.min()
        hi = t_uni.max()
    span = (hi - lo).clamp_min(1e-08)
    eps = 1e-06 * span
    if bool((span <= 2.0 * eps).item()):
        return torch.full(
            (n_points,), float(0.5 * (lo + hi)), device=device, dtype=t_uni.dtype
        )
    return (
        lo
        + eps
        + torch.rand(n_points, device=device, dtype=t_uni.dtype) * (span - 2.0 * eps)
    )


def _observed_spatial_mask_from_sample_mask(mask_sample):
    mask_bool = torch.as_tensor(mask_sample) > 0
    if mask_bool.dim() == 4:
        mask_bool = torch.any(mask_bool, dim=0)
        mask_bool = torch.any(mask_bool, dim=-1)
    elif mask_bool.dim() == 3:
        mask_bool = torch.any(mask_bool, dim=0)
    elif mask_bool.dim() != 2:
        raise ValueError(
            f"Expected sample mask shape (T,H,W[,C]) or (H,W), got {tuple(mask_bool.shape)}"
        )
    return np.asarray(mask_bool.detach().cpu().numpy(), dtype=bool)


def _candidate_spatial_mask_from_sample_mask(mask_sample, sample_mode):
    sample_mode = _normalize_pde_point_sample_mode(sample_mode)
    observed_mask = _observed_spatial_mask_from_sample_mask(mask_sample)
    if sample_mode in ("observed_grid",):
        return observed_mask
    if sample_mode in ("unobserved_grid",):
        return ~observed_mask
    raise ValueError(f"Unsupported candidate-mask sample_mode={sample_mode}")


def _select_valid_spatial_indices(
    valid_mask, sample_ratio, max_points, total_spatial_count=None, rng=None
):
    valid_idx = np.argwhere(np.asarray(valid_mask, dtype=bool))
    if valid_idx.size == 0:
        return None
    valid_count = int(valid_idx.shape[0])
    total_spatial_count = int(
        valid_count if total_spatial_count is None else total_spatial_count
    )
    sample_ratio = float(sample_ratio)
    max_points = int(max_points or 0)
    if max_points > 0:
        n_points = min(max_points, valid_count)
    else:
        n_points = min(
            max(1, int(math.ceil(total_spatial_count * max(sample_ratio, 0.0)))),
            valid_count,
        )
    if n_points <= 0:
        return None
    if n_points < valid_count:
        chosen_idx = (
            np.random.choice(valid_count, size=n_points, replace=False)
            if rng is None
            else rng.choice(valid_count, size=n_points, replace=False)
        )
        return valid_idx[chosen_idx]
    return valid_idx


def _sample_mask_conditioned_spatial_coords_for_sample(
    *,
    coord_uniques,
    mask_sample,
    sample_ratio,
    max_points,
    device,
    sample_mode,
    core_parameterization="per_time",
    seed_text=None,
):
    sample_mode = _normalize_pde_point_sample_mode(sample_mode)
    if sample_mode == "continuous":
        return _sample_continuous_spatial_points(
            coord_uniques,
            sample_ratio,
            max_points,
            device,
            core_parameterization="per_time",
        )
    seed = (
        None
        if seed_text is None
        else int(hashlib.sha1(str(seed_text).encode("utf-8")).hexdigest()[:8], 16)
    )
    rng = None if seed is None else np.random.default_rng(seed)
    valid_mask = _candidate_spatial_mask_from_sample_mask(mask_sample, sample_mode)
    spatial_u, spatial_v = _spatial_coord_uniques(
        coord_uniques, core_parameterization="per_time"
    )
    logical_u = (
        torch.as_tensor(spatial_u, dtype=torch.float32)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    logical_v = (
        torch.as_tensor(spatial_v, dtype=torch.float32)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    total_spatial_count = int(logical_u.size) * int(logical_v.size)
    chosen = _select_valid_spatial_indices(
        valid_mask,
        sample_ratio,
        max_points,
        total_spatial_count=total_spatial_count,
        rng=rng,
    )
    if chosen is None:
        return None
    sample_coords = np.stack(
        [logical_u[chosen[:, 0]], logical_v[chosen[:, 1]]], axis=-1
    ).astype(np.float32, copy=False)
    return torch.as_tensor(sample_coords, dtype=torch.float32, device=device)


def _observed_spatial_mask_3d(mask_sample, spatial_shape):
    """Union of sensors across time/channels, in the tensor's spatial axis order."""
    observed = torch.as_tensor(mask_sample) > 0
    if observed.dim() == 5:
        observed = observed.any(dim=0).any(dim=-1)
    elif observed.dim() == 4:
        observed = observed.any(dim=0)
    elif observed.dim() != 3:
        raise ValueError(f"Expected (T,X,Y,Z[,C]) mask, got {tuple(observed.shape)}")
    observed = np.asarray(observed.detach().cpu().numpy(), dtype=bool)
    return np.broadcast_to(observed, tuple(spatial_shape))


def _sample_mask_conditioned_spatial_coords_3d(
    *,
    coord_uniques,
    mask_sample,
    sample_ratio,
    max_points,
    device,
    sample_mode,
    seed_text=None,
    interior_margins=None,
):
    axes = [
        torch.as_tensor(a).detach().cpu().numpy().reshape(-1) for a in coord_uniques
    ]
    if len(axes) != 3:
        raise ValueError("3D conditioned sampling requires three spatial axes")
    mode = _normalize_pde_point_sample_mode(sample_mode)
    observed = _observed_spatial_mask_3d(mask_sample, [len(a) for a in axes])
    valid = observed.copy() if mode.startswith("observed_") else ~observed
    if interior_margins is not None:
        for axis, margin in enumerate(interior_margins):
            eligible = (axes[axis] >= axes[axis][0] + margin - 1e-07) & (
                axes[axis] <= axes[axis][-1] - margin + 1e-07
            )
            axis_shape = [1, 1, 1]
            axis_shape[axis] = len(eligible)
            valid &= eligible.reshape(axis_shape)
    if mode not in ("observed_grid", "unobserved_grid"):
        raise ValueError(f"Unsupported 3D mask-conditioned mode: {mode}")
    seed = (
        None
        if seed_text is None
        else int(hashlib.sha1(str(seed_text).encode()).hexdigest()[:8], 16)
    )
    rng = None if seed is None else np.random.default_rng(seed)
    chosen = _select_valid_spatial_indices(
        valid,
        sample_ratio,
        max_points,
        total_spatial_count=int(np.prod(valid.shape)),
        rng=rng,
    )
    if chosen is None:
        return None
    coords = np.stack([axes[i][chosen[:, i]] for i in range(3)], axis=-1)
    return torch.as_tensor(coords, dtype=torch.float32, device=device)


def _compute_mask_conditioned_3d_pde_loss(
    *,
    model_ref,
    core_batch,
    coord_uniques,
    norm_stats,
    dataset_type,
    pde_kwargs,
    channel_order,
    sample_ratio,
    max_pde_points,
    shared_basis_multichannel,
    mask_batch,
    backward_scale=0.0,
):
    if mask_batch is None:
        raise ValueError("Mask-conditioned 3D PDE sampling requires mask_batch")
    mode = _normalize_pde_point_sample_mode(pde_kwargs.get("point_sample_mode"))
    accum, used = ({}, 0)
    sampled = []
    batch_count = int(core_batch.shape[0])
    for i in range(batch_count):
        if dataset_type == "ocean":
            shape = [int(torch.as_tensor(a).numel()) for a in coord_uniques]
            horizontal = _observed_spatial_mask_3d(mask_batch[i], shape).any(axis=0)
            coords = _sample_mask_conditioned_spatial_coords_for_sample(
                coord_uniques=coord_uniques[1:],
                mask_sample=horizontal,
                sample_ratio=sample_ratio,
                max_points=max_pde_points,
                device=core_batch.device,
                sample_mode=mode,
            )
        else:
            from physftm.pde.query_fd import _physical_to_logical_delta

            margins = [
                _physical_to_logical_delta(
                    pde_kwargs.get(fd), axis, pde_kwargs[length], False
                )[1]
                for axis, fd, length in zip(
                    coord_uniques, ("fd_dx", "fd_dy", "fd_dz"), ("Lx", "Ly", "Lz")
                )
            ]
            coords = _sample_mask_conditioned_spatial_coords_3d(
                coord_uniques=coord_uniques,
                mask_sample=mask_batch[i],
                sample_ratio=sample_ratio,
                max_points=max_pde_points,
                device=core_batch.device,
                sample_mode=mode,
                interior_margins=margins,
            )
        if coords is None:
            continue
        sampled.append((i, coords))
    if not sampled:
        return None
    for i, coords in sampled:
        terms = _compute_continuous_sampled_pde_loss(
            model_ref=model_ref,
            core_batch=core_batch[i : i + 1],
            coord_uniques=coord_uniques,
            norm_stats=norm_stats,
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
            channel_order=channel_order,
            sample_ratio=sample_ratio,
            max_pde_points=max_pde_points,
            shared_basis_multichannel=shared_basis_multichannel,
            backward_scale=backward_scale / len(sampled),
            sample_coords_override=coords,
        )
        if terms is not None:
            used += 1
            for key, value in terms.items():
                if key == "_pde_grad_accumulated":
                    accum[key] = bool(value)
                else:
                    accum[key] = accum.get(key, 0) + value
    if not used:
        return None
    return {
        k: v if k == "_pde_grad_accumulated" else v / used for k, v in accum.items()
    }


def _sample_core_with_mode_basis_2d(core_batch, U, V, shared_basis_multichannel=False):
    if shared_basis_multichannel:
        tmp = torch.einsum("pi,btcij->btcpj", U, core_batch)
        out = torch.einsum("pj,btcpj->btcp", V, tmp)
        return out.permute(0, 1, 3, 2).contiguous()
    if core_batch.dim() != 4:
        raise ValueError(
            f"Expected scalar 2D core with dim=4, got shape={tuple(core_batch.shape)}"
        )
    tmp = torch.einsum("pi,btij->btpj", U, core_batch)
    out = torch.einsum("pj,btpj->btp", V, tmp)
    return out.unsqueeze(-1)


def _sample_field_with_mode_basis_2d(
    basis_function,
    core_batch,
    sample_coords,
    coord_uniques=None,
    shared_basis_multichannel=False,
):
    sample_coords = torch.as_tensor(
        sample_coords, dtype=core_batch.dtype, device=core_batch.device
    )
    U = basis_function.mode_u(sample_coords[:, 0])
    V = basis_function.mode_v(sample_coords[:, 1])
    return _sample_core_with_mode_basis_2d(
        core_batch, U, V, shared_basis_multichannel=shared_basis_multichannel
    )


def _compute_ac_offgrid_point_residual_sq(
    *, query_fn, sample_coords, coord_uniques, pde_kwargs, return_terms=False
):
    from physftm.pde.query_fd import _compute_ac_fd_query_terms

    time_step = _pde_time_step(pde_kwargs, "per_time")
    spatial_u, spatial_v = _spatial_coord_uniques(
        coord_uniques, core_parameterization="per_time"
    )
    terms = _compute_ac_fd_query_terms(
        query_fn=query_fn,
        sample_coords=sample_coords,
        coord_t=None,
        coord_u=spatial_u,
        coord_v=spatial_v,
        dt=time_step,
        Lx=pde_kwargs["Lx"],
        Ly=pde_kwargs["Ly"],
        ac_lambda=pde_kwargs["ac_lambda"],
        epsilon=pde_kwargs["epsilon"],
        fd_dx=pde_kwargs.get("fd_dx", None),
        fd_dy=pde_kwargs.get("fd_dy", None),
        fd_space_order=pde_kwargs.get("fd_space_order", 2),
        fd_time_order=pde_kwargs.get("fd_time_order", 2),
        query_time_continuous=False,
    )
    if return_terms:
        return {
            "residual_sq": terms["residual"] ** 2,
            "residual": terms["residual"],
            "u_t": terms["u_t"],
            "diffusion": terms["diffusion"],
            "nonlinear": -terms["nonlinear"],
        }
    return terms["residual"] ** 2


def _compute_offgrid_point_residual_terms(
    *, query_fn, sample_coords, sample_t_coords, coord_uniques, pde_kwargs
):
    dataset_type = str(pde_kwargs.get("dataset_type", "ac")).lower()
    time_step = _pde_time_step(pde_kwargs, "per_time")
    spatial_u, spatial_v = _spatial_coord_uniques(
        coord_uniques, core_parameterization="per_time"
    )
    if dataset_type == "ac":
        return _compute_ac_offgrid_point_residual_sq(
            query_fn=query_fn,
            sample_coords=sample_coords,
            coord_uniques=coord_uniques,
            pde_kwargs=pde_kwargs,
            return_terms=True,
        )
    if dataset_type == "ns":
        from physftm.pde.query_fd import compute_ns_pde_loss_fd_query

        return compute_ns_pde_loss_fd_query(
            query_fn=query_fn,
            sample_coords=sample_coords,
            sample_t_coords=sample_t_coords,
            coord_t=None,
            coord_u=spatial_u,
            coord_v=spatial_v,
            dt=time_step,
            Lx=pde_kwargs["Lx"],
            Ly=pde_kwargs["Ly"],
            nu=pde_kwargs["nu"],
            forcing_scale=pde_kwargs["forcing_scale"],
            channel_order=pde_kwargs.get("channel_order"),
            fd_dx=pde_kwargs.get("fd_dx", None),
            fd_dy=pde_kwargs.get("fd_dy", None),
            fd_space_order=pde_kwargs.get("fd_space_order", 2),
            fd_time_order=pde_kwargs.get("fd_time_order", 2),
            query_time_continuous=False,
            return_terms=True,
        )
    raise ValueError(f"Unsupported PDE residual terms dataset_type={dataset_type}")


def _slice_sample_time_coords_for_chunk(sample_time_coords, start, end, total_points):
    if sample_time_coords is None:
        return None
    if torch.is_tensor(sample_time_coords):
        if sample_time_coords.ndim > 0 and int(sample_time_coords.shape[0]) == int(
            total_points
        ):
            return sample_time_coords[start:end]
        return sample_time_coords
    arr = np.asarray(sample_time_coords)
    if arr.ndim > 0 and int(arr.shape[0]) == int(total_points):
        return arr[start:end]
    return sample_time_coords


def _merge_weighted_query_loss(accum, chunk_losses, weight):
    for key, value in chunk_losses.items():
        if not torch.is_tensor(value):
            value = torch.as_tensor(float(value), dtype=torch.float32)
        if key.startswith("fd_"):
            if key not in accum:
                accum[key] = value
            continue
        weighted = value * float(weight)
        accum[key] = weighted if key not in accum else accum[key] + weighted


def _finalize_weighted_query_loss(accum, total_weight):
    if total_weight <= 0:
        return None
    return {
        key: value if key.startswith("fd_") else value / float(total_weight)
        for key, value in accum.items()
    }


def _compute_per_time_ac_factor_pde_terms_chunk(
    *,
    basis_function,
    core_batch,
    sample_coords,
    coord_uniques,
    affine_scale,
    affine_bias,
    pde_kwargs,
):
    from physftm.pde.query_fd import (
        _FDQueryField2D,
        _center_time_slice_ordered,
        _temporal_central_diff_ordered,
        _validate_fd_order,
    )

    fd_space_order = _validate_fd_order(
        pde_kwargs.get("fd_space_order", 2), name="fd_space_order"
    )
    fd_time_order = _validate_fd_order(
        pde_kwargs.get("fd_time_order", 2), name="fd_time_order"
    )
    max_offset = 1 if fd_space_order == 2 else 2
    spatial_u, spatial_v = _spatial_coord_uniques(
        coord_uniques, core_parameterization="per_time"
    )
    sampler = _FDQueryField2D(
        query_fn=None,
        sample_coords=sample_coords,
        coord_u=spatial_u,
        coord_v=spatial_v,
        physical_length_u=pde_kwargs["Lx"],
        physical_length_v=pde_kwargs["Ly"],
        periodic_u=True,
        periodic_v=True,
        fd_du=pde_kwargs.get("fd_dx", None),
        fd_dv=pde_kwargs.get("fd_dy", None),
        max_offset_u=max_offset,
        max_offset_v=max_offset,
    )
    center_coords = sampler.shifted_coords()
    U0 = basis_function.mode_u(center_coords[:, 0])
    V0 = basis_function.mode_v(center_coords[:, 1])

    def _factor_at_offset(mode, axis, offset):
        coords = (
            sampler.shifted_coords(offset, 0.0)
            if axis == 0
            else sampler.shifted_coords(0.0, offset)
        )
        return mode(coords[:, axis])

    def _second_derivative_factor(mode, axis, center, step):
        if fd_space_order == 2:
            return (
                _factor_at_offset(mode, axis, +1.0)
                - 2.0 * center
                + _factor_at_offset(mode, axis, -1.0)
            ) / float(step) ** 2
        return (
            -_factor_at_offset(mode, axis, +2.0)
            + 16.0 * _factor_at_offset(mode, axis, +1.0)
            - 30.0 * center
            + 16.0 * _factor_at_offset(mode, axis, -1.0)
            - _factor_at_offset(mode, axis, -2.0)
        ) / (12.0 * float(step) ** 2)

    Uxx = _second_derivative_factor(basis_function.mode_u, 0, U0, sampler.du_phys)
    Vyy = _second_derivative_factor(basis_function.mode_v, 1, V0, sampler.dv_phys)
    u_norm = _sample_core_with_mode_basis_2d(
        core_batch, U0, V0, shared_basis_multichannel=False
    )[..., 0]
    u_xx_norm = _sample_core_with_mode_basis_2d(
        core_batch, Uxx, V0, shared_basis_multichannel=False
    )[..., 0]
    u_yy_norm = _sample_core_with_mode_basis_2d(
        core_batch, U0, Vyy, shared_basis_multichannel=False
    )[..., 0]
    scale = affine_scale[..., 0] if int(affine_scale.shape[-1]) == 1 else affine_scale
    bias = affine_bias[..., 0] if int(affine_bias.shape[-1]) == 1 else affine_bias
    u = u_norm * scale + bias
    lap_all = (u_xx_norm + u_yy_norm) * scale
    time_step = _pde_time_step(pde_kwargs, "per_time")
    u_t = _temporal_central_diff_ordered(u, time_step, fd_time_order)
    u_c = _center_time_slice_ordered(u, fd_time_order)
    lap = _center_time_slice_ordered(lap_all, fd_time_order)
    diffusion = float(pde_kwargs["ac_lambda"]) * float(pde_kwargs["epsilon"]) ** 2 * lap
    nonlinear = float(pde_kwargs["ac_lambda"]) * (u_c**3 - u_c)
    residual = u_t - diffusion + nonlinear
    return {
        "residual_sq": residual**2,
        "residual": residual,
        "u_t": u_t,
        "diffusion": diffusion,
        "nonlinear": -nonlinear,
        "lap": lap,
        "fd_dx": torch.as_tensor(sampler.du_phys, dtype=u.dtype, device=u.device),
        "fd_dy": torch.as_tensor(sampler.dv_phys, dtype=u.dtype, device=u.device),
    }


def _compute_per_time_ac_factor_pde_loss_chunk(
    *,
    basis_function,
    core_batch,
    sample_coords,
    coord_uniques,
    affine_scale,
    affine_bias,
    pde_kwargs,
):
    terms = _compute_per_time_ac_factor_pde_terms_chunk(
        basis_function=basis_function,
        core_batch=core_batch,
        sample_coords=sample_coords,
        coord_uniques=coord_uniques,
        affine_scale=affine_scale,
        affine_bias=affine_bias,
        pde_kwargs=pde_kwargs,
    )

    def _mse(value):
        return torch.mean(value**2)

    return {
        "pde": _mse(terms["residual"]),
        "term_residual": _mse(terms["residual"]),
        "term_u_t": _mse(terms["u_t"]),
        "term_diffusion": _mse(terms["diffusion"]),
        "term_lap": _mse(terms["lap"]),
        "fd_dx": terms["fd_dx"],
        "fd_dy": terms["fd_dy"],
    }


def _compute_per_time_ac_factor_pde_loss(
    *,
    model_ref,
    core_batch,
    sample_coords,
    coord_uniques,
    norm_stats,
    pde_kwargs,
    backward_scale=0.0,
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
    should_stream = (
        configured_chunk_size > 0
        and total_points > configured_chunk_size
        and (float(backward_scale) != 0.0 or not torch.is_grad_enabled())
    )
    if not should_stream:
        return _compute_per_time_ac_factor_pde_loss_chunk(
            basis_function=model_ref.basis_function,
            core_batch=core_batch,
            sample_coords=sample_coords,
            coord_uniques=coord_uniques,
            affine_scale=affine_scale,
            affine_bias=affine_bias,
            pde_kwargs=pde_kwargs,
        )
    chunk_size = min(configured_chunk_size, total_points)
    accum = {}
    total_weight = 0
    backwarded = False
    for start in range(0, total_points, chunk_size):
        end = min(start + chunk_size, total_points)
        chunk_losses = _compute_per_time_ac_factor_pde_loss_chunk(
            basis_function=model_ref.basis_function,
            core_batch=core_batch,
            sample_coords=sample_coords[start:end],
            coord_uniques=coord_uniques,
            affine_scale=affine_scale,
            affine_bias=affine_bias,
            pde_kwargs=pde_kwargs,
        )
        weight = int(end - start)
        if float(backward_scale) != 0.0:
            (
                chunk_losses["pde"]
                * (float(backward_scale) * float(weight) / float(total_points))
            ).backward()
            backwarded = True
        detached_losses = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in chunk_losses.items()
        }
        _merge_weighted_query_loss(accum, detached_losses, weight)
        total_weight += weight
        del chunk_losses
    finalized = _finalize_weighted_query_loss(accum, total_weight)
    if finalized is not None and backwarded:
        finalized["_pde_grad_accumulated"] = True
    return finalized


def _compute_query_sampled_pde_loss(
    *,
    query_fn,
    sample_coords,
    sample_time_coords,
    coord_uniques,
    core_parameterization,
    dataset_type,
    pde_kwargs,
    channel_order,
    backward_scale=0.0,
):
    dataset_type = str(dataset_type).lower()
    query_time_continuous = False
    coord_t = coord_uniques[0] if query_time_continuous else None
    time_step = _pde_time_step(pde_kwargs, "per_time")
    spatial_u, spatial_v = _spatial_coord_uniques(
        coord_uniques, core_parameterization="per_time"
    )
    query_chunk_size = int(pde_kwargs.get("query_chunk_size", 0) or 0)
    sample_count = (
        int(sample_coords.shape[0])
        if torch.is_tensor(sample_coords) and sample_coords.ndim > 0
        else 0
    )
    can_release_chunk_graphs = (
        float(backward_scale) != 0.0 or not torch.is_grad_enabled()
    )
    if (
        query_chunk_size > 0
        and sample_count > query_chunk_size
        and can_release_chunk_graphs
    ):
        no_recursive_chunk_kwargs = dict(pde_kwargs)
        no_recursive_chunk_kwargs["query_chunk_size"] = 0
        accum = {}
        total_weight = 0
        backwarded = False
        for start in range(0, sample_count, query_chunk_size):
            end = min(start + query_chunk_size, sample_count)
            chunk_losses = _compute_query_sampled_pde_loss(
                query_fn=query_fn,
                sample_coords=sample_coords[start:end],
                sample_time_coords=_slice_sample_time_coords_for_chunk(
                    sample_time_coords, start, end, sample_count
                ),
                coord_uniques=coord_uniques,
                core_parameterization="per_time",
                dataset_type=dataset_type,
                pde_kwargs=no_recursive_chunk_kwargs,
                channel_order=channel_order,
                backward_scale=0.0,
            )
            if chunk_losses is None:
                continue
            weight = int(end - start)
            if float(backward_scale) != 0.0 and "pde" in chunk_losses:
                (
                    chunk_losses["pde"]
                    * (float(backward_scale) * float(weight) / float(sample_count))
                ).backward()
                backwarded = True
            detached_losses = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in chunk_losses.items()
            }
            _merge_weighted_query_loss(accum, detached_losses, weight)
            total_weight += weight
            del chunk_losses
        finalized = _finalize_weighted_query_loss(accum, total_weight)
        if finalized is not None and backwarded:
            finalized["_pde_grad_accumulated"] = True
        return finalized
    if dataset_type == "ac":
        from physftm.pde.query_fd import compute_ac_pde_loss_fd_query

        return compute_ac_pde_loss_fd_query(
            query_fn=query_fn,
            sample_coords=sample_coords,
            sample_t_coords=sample_time_coords,
            coord_t=coord_t,
            coord_u=spatial_u,
            coord_v=spatial_v,
            dt=time_step,
            Lx=pde_kwargs["Lx"],
            Ly=pde_kwargs["Ly"],
            ac_lambda=pde_kwargs["ac_lambda"],
            epsilon=pde_kwargs["epsilon"],
            fd_dx=pde_kwargs.get("fd_dx", None),
            fd_dy=pde_kwargs.get("fd_dy", None),
            fd_space_order=pde_kwargs.get("fd_space_order", 2),
            fd_time_order=pde_kwargs.get("fd_time_order", 2),
            query_time_continuous=query_time_continuous,
        )
    if channel_order is None:
        raise ValueError(
            f"channel_order is required for fd_query PDE loss on dataset_type={dataset_type}"
        )
    if dataset_type == "ns":
        from physftm.pde.query_fd import compute_ns_pde_loss_fd_query

        return compute_ns_pde_loss_fd_query(
            query_fn=query_fn,
            sample_coords=sample_coords,
            sample_t_coords=sample_time_coords,
            coord_t=coord_t,
            coord_u=spatial_u,
            coord_v=spatial_v,
            dt=time_step,
            Lx=pde_kwargs["Lx"],
            Ly=pde_kwargs["Ly"],
            nu=pde_kwargs["nu"],
            forcing_scale=pde_kwargs["forcing_scale"],
            channel_order=channel_order,
            fd_dx=pde_kwargs.get("fd_dx", None),
            fd_dy=pde_kwargs.get("fd_dy", None),
            fd_space_order=pde_kwargs.get("fd_space_order", 2),
            fd_time_order=pde_kwargs.get("fd_time_order", 2),
            query_time_continuous=query_time_continuous,
        )
    return None


def _compute_continuous_sampled_pde_loss(
    *,
    model_ref,
    core_batch,
    coord_uniques,
    norm_stats,
    dataset_type,
    pde_kwargs,
    channel_order,
    sample_ratio,
    max_pde_points,
    shared_basis_multichannel,
    batch_idx=None,
    pde_context=None,
    backward_scale=0.0,
    sample_coords_override=None,
):
    dataset_type = str(dataset_type).lower()
    if dataset_type == "heat3d":
        if len(coord_uniques) != 3:
            raise ValueError("Heat3D requires three spatial coordinate axes")
        sample_coords = (
            sample_coords_override
            if sample_coords_override is not None
            else _sample_continuous_spatial_points(
                coord_uniques,
                sample_ratio=sample_ratio,
                max_points=max_pde_points,
                device=core_batch.device,
                core_parameterization="per_time",
                allow_full_sample=True,
                spatial_axis_order=("u", "v", "w"),
            )
        )
        if sample_coords is None:
            return None
        num_channels = _num_channels_from_core(
            core_batch,
            shared_basis_multichannel=shared_basis_multichannel,
            core_parameterization="per_time",
        )
        affine_scale, affine_bias = _inverse_norm_affine_channel_last(
            norm_stats,
            num_channels=num_channels,
            field_ndim=3,
            device=core_batch.device,
            dtype=core_batch.dtype,
        )
        from physftm.pde.query_fd import (
            _sample_field_with_mode_basis_3d,
            compute_heat3d_pde_loss_fd_query,
        )

        def query_fn(query_coords):
            sampled = _sample_field_with_mode_basis_3d(
                model_ref.basis_function,
                core_batch,
                query_coords,
                spatial_axis_order=("u", "v", "w"),
                shared_basis_multichannel=shared_basis_multichannel,
            )
            return sampled * affine_scale + affine_bias

        return compute_heat3d_pde_loss_fd_query(
            query_fn=query_fn,
            sample_coords=sample_coords,
            coord_u=coord_uniques[0],
            coord_v=coord_uniques[1],
            coord_w=coord_uniques[2],
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
    sample_coords = _sample_continuous_spatial_points(
        coord_uniques,
        sample_ratio=sample_ratio,
        max_points=max_pde_points,
        device=core_batch.device,
        core_parameterization="per_time",
    )
    sample_time_coords = _sample_continuous_time_points(
        coord_uniques,
        sample_ratio=sample_ratio,
        max_points=max_pde_points,
        device=core_batch.device,
        core_parameterization="per_time",
    )
    if sample_coords is None:
        return None
    if (
        "per_time" == "per_time"
        and dataset_type == "ac"
        and (not shared_basis_multichannel)
    ):
        return _compute_per_time_ac_factor_pde_loss(
            model_ref=model_ref,
            core_batch=core_batch,
            sample_coords=sample_coords,
            coord_uniques=coord_uniques,
            norm_stats=norm_stats,
            pde_kwargs=pde_kwargs,
            backward_scale=backward_scale,
        )
    num_channels = _num_channels_from_core(
        core_batch,
        shared_basis_multichannel=shared_basis_multichannel,
        core_parameterization="per_time",
    )
    affine_scale, affine_bias = _inverse_norm_affine_channel_last(
        norm_stats,
        num_channels=num_channels,
        field_ndim=3,
        device=core_batch.device,
        dtype=core_batch.dtype,
    )

    def query_fn(query_coords):
        sampled = _sample_field_with_mode_basis_2d(
            model_ref.basis_function,
            core_batch,
            query_coords,
            coord_uniques=coord_uniques,
            shared_basis_multichannel=shared_basis_multichannel,
        )
        return sampled * affine_scale + affine_bias

    return _compute_query_sampled_pde_loss(
        query_fn=query_fn,
        sample_coords=sample_coords,
        sample_time_coords=sample_time_coords,
        coord_uniques=coord_uniques,
        core_parameterization="per_time",
        dataset_type=dataset_type,
        pde_kwargs=pde_kwargs,
        channel_order=channel_order,
        backward_scale=backward_scale,
    )


def _compute_mask_conditioned_sampled_pde_loss(
    *,
    model_ref,
    core_batch,
    coord_uniques,
    norm_stats,
    dataset_type,
    pde_kwargs,
    channel_order,
    sample_ratio,
    max_pde_points,
    shared_basis_multichannel,
    mask_batch,
):
    if mask_batch is None:
        return None
    num_channels = _num_channels_from_core(
        core_batch,
        shared_basis_multichannel=shared_basis_multichannel,
        core_parameterization="per_time",
    )
    affine_scale, affine_bias = _inverse_norm_affine_channel_last(
        norm_stats,
        num_channels=num_channels,
        field_ndim=3,
        device=core_batch.device,
        dtype=core_batch.dtype,
    )
    sample_mode = _normalize_pde_point_sample_mode(
        pde_kwargs.get("point_sample_mode", "continuous")
    )
    accum = {}
    used_samples = 0
    for local_idx in range(int(core_batch.shape[0])):
        sample_coords = _sample_mask_conditioned_spatial_coords_for_sample(
            coord_uniques=coord_uniques,
            mask_sample=mask_batch[local_idx],
            sample_ratio=sample_ratio,
            max_points=max_pde_points,
            device=core_batch.device,
            sample_mode=sample_mode,
            core_parameterization="per_time",
        )
        if sample_coords is None:
            continue
        sample_time_coords = _sample_continuous_time_points(
            coord_uniques,
            sample_ratio=sample_ratio,
            max_points=max_pde_points,
            device=core_batch.device,
            core_parameterization="per_time",
        )

        def query_fn(query_coords):
            sampled = _sample_field_with_mode_basis_2d(
                model_ref.basis_function,
                core_batch[local_idx : local_idx + 1],
                query_coords,
                coord_uniques=coord_uniques,
                shared_basis_multichannel=shared_basis_multichannel,
            )
            return sampled * affine_scale + affine_bias

        sample_losses = _compute_query_sampled_pde_loss(
            query_fn=query_fn,
            sample_coords=sample_coords,
            sample_time_coords=sample_time_coords,
            coord_uniques=coord_uniques,
            core_parameterization="per_time",
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
            channel_order=channel_order,
        )
        if sample_losses is None:
            continue
        for key, value in sample_losses.items():
            if torch.is_tensor(value):
                accum[key] = value if key not in accum else accum[key] + value
        used_samples += 1
    if used_samples <= 0:
        return None
    return {key: value / float(used_samples) for key, value in accum.items()}


def compute_pde_loss_for_training(
    *,
    model_ref,
    core_batch,
    dense_output,
    norm_stats,
    dataset_type,
    pde_kwargs,
    channel_order,
    point_sample_ratio,
    point_max_points,
    coord_uniques,
    shared_basis_multichannel,
    lambda_pde_point,
    compute_pde_point=None,
    batch_idx=None,
    pde_context=None,
    mask_batch=None,
    backward_pde=False,
):
    losses = {}
    dataset_type = str(dataset_type).lower()
    compute_pde_point = (
        float(lambda_pde_point) > 0.0
        if compute_pde_point is None
        else bool(compute_pde_point)
    )
    point_sample_mode = _normalize_pde_point_sample_mode(
        pde_kwargs.get("point_sample_mode", "continuous")
    )
    if dataset_type == "ocean":
        if (
            "per_time" != "per_time"
            or len(coord_uniques) != 3
            or (not shared_basis_multichannel)
        ):
            raise ValueError(
                "Ocean equation-free loss requires multichannel per-time 3D data."
            )
        sampler = (
            _compute_continuous_sampled_pde_loss
            if point_sample_mode == "continuous"
            else _compute_mask_conditioned_3d_pde_loss
        )
        sampling_extra = (
            {} if point_sample_mode == "continuous" else {"mask_batch": mask_batch}
        )
        sampled_losses = sampler(
            model_ref=model_ref,
            core_batch=core_batch,
            coord_uniques=coord_uniques,
            norm_stats=norm_stats,
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
            channel_order=channel_order,
            sample_ratio=float(
                pde_kwargs.get("spatial_sample_ratio", point_sample_ratio)
            ),
            max_pde_points=int(
                pde_kwargs.get("spatial_max_points", point_max_points) or 0
            ),
            shared_basis_multichannel=shared_basis_multichannel,
            backward_scale=float(lambda_pde_point) if bool(backward_pde) else 0.0,
            **sampling_extra,
        )
        if sampled_losses is None:
            return losses
        losses["pde_point"] = sampled_losses["pde"]
        for sampled_name, sampled_val in sampled_losses.items():
            if sampled_name.startswith("term_"):
                losses[f"pde_point_{sampled_name[5:]}"] = sampled_val
        if sampled_losses.get("_pde_grad_accumulated", False):
            losses["_pde_grad_accumulated"] = True
        return losses
    if dataset_type == "heat3d":
        if (
            "per_time" != "per_time"
            or len(coord_uniques) != 3
            or shared_basis_multichannel
        ):
            raise ValueError("Heat3D PDE requires scalar per-time 3D RKHS data.")
        sampler = (
            _compute_continuous_sampled_pde_loss
            if point_sample_mode == "continuous"
            else _compute_mask_conditioned_3d_pde_loss
        )
        sampling_extra = (
            {} if point_sample_mode == "continuous" else {"mask_batch": mask_batch}
        )
        sampled_losses = sampler(
            model_ref=model_ref,
            core_batch=core_batch,
            coord_uniques=coord_uniques,
            norm_stats=norm_stats,
            dataset_type=dataset_type,
            pde_kwargs=pde_kwargs,
            channel_order=channel_order,
            sample_ratio=point_sample_ratio,
            max_pde_points=point_max_points,
            shared_basis_multichannel=shared_basis_multichannel,
            backward_scale=float(lambda_pde_point) if bool(backward_pde) else 0.0,
            **sampling_extra,
        )
        if sampled_losses is None:
            return losses
        losses["pde_point"] = sampled_losses["pde"]
        for sampled_name, sampled_val in sampled_losses.items():
            if sampled_name.startswith("term_"):
                losses[f"pde_point_{sampled_name[5:]}"] = sampled_val
        return losses
    if dataset_type == "ns":
        ns_query_multichannel = (
            bool(shared_basis_multichannel)
            and channel_order is not None
            and all(
                (
                    name in tuple((str(x) for x in channel_order))
                    for name in ("u", "v", "omega")
                )
            )
        )
        if not ns_query_multichannel:
            raise ValueError("NS PDE training requires the (u,v,omega) channels.")
    use_continuous_points = (
        compute_pde_point
        and (
            point_sample_mode != "continuous"
            or float(point_sample_ratio) < 1.0
            or int(point_max_points or 0) > 0
        )
        and hasattr(model_ref.basis_function, "mode_u")
        and hasattr(model_ref.basis_function, "mode_v")
        and (
            dataset_type == "heat3d"
            or len(coord_uniques) == 2
            or (
                len(coord_uniques) > 2
                and int(torch.as_tensor(coord_uniques[2]).numel()) == 1
            )
        )
    )
    if use_continuous_points:
        if point_sample_mode == "continuous":
            sampled_losses = _compute_continuous_sampled_pde_loss(
                model_ref=model_ref,
                core_batch=core_batch,
                coord_uniques=coord_uniques,
                norm_stats=norm_stats,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=channel_order,
                sample_ratio=point_sample_ratio,
                max_pde_points=point_max_points,
                shared_basis_multichannel=shared_basis_multichannel,
                batch_idx=batch_idx,
                pde_context=pde_context,
                backward_scale=float(lambda_pde_point) if bool(backward_pde) else 0.0,
            )
        else:
            sampled_losses = _compute_mask_conditioned_sampled_pde_loss(
                model_ref=model_ref,
                core_batch=core_batch,
                coord_uniques=coord_uniques,
                norm_stats=norm_stats,
                dataset_type=dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=channel_order,
                sample_ratio=point_sample_ratio,
                max_pde_points=point_max_points,
                shared_basis_multichannel=shared_basis_multichannel,
                mask_batch=mask_batch,
            )
        if sampled_losses is not None and "pde" in sampled_losses:
            losses["pde_point"] = sampled_losses["pde"]
            for sampled_name, sampled_val in sampled_losses.items():
                if sampled_name.startswith("term_"):
                    losses[f"pde_point_{sampled_name[5:]}"] = sampled_val
    return losses
