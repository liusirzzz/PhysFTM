import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
import torch
from physftm.artifacts.loading import (
    inverse_normalize_tensor,
    load_basis_and_coords,
    load_core_payload,
)
from physftm.data.axes import spatial_axis_order_from_manifest
from physftm.data.loading import get_mask_batch
from physftm.data.normalization import apply_saved_normalization
from physftm.data.splits import resolve_split_paths
from physftm.inference.config import _parse_args
from physftm.inference.core import (
    _as_core_train_array,
    _check_core_compatible,
    _decode_with_precomputed_basis,
    _mean_prior_core_batch,
    _query_coord_uniques_for_inference,
)
from physftm.inference.io import (
    RKHSInferenceResult,
    _build_results_dir,
    _crop_time_axis_to_core_if_requested,
    _ensure_5d,
    _infer_channel_names,
    _load_large_data_flexible,
    _parse_plot_time_indices,
    _resolve_checkpoint_paths,
    _sync_device_if_needed,
)
from physftm.inference.metrics import (
    _accumulate_channel_metrics,
    _finalize_channel_metrics,
    _plot_sample,
    compute_ssim_metrics_batched,
)
from physftm.inference.pde_eval import (
    _build_pde_kwargs,
    _compute_ftm_native_query_pde_batched,
    _resolve_pde_dataset_type,
)
from physftm.pde.training import (
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


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def run_rkhs_split_inference(args):
    if not getattr(args, "ftm_manifest_path", None):
        raise ValueError(
            "Inference config must set artifact.ftm_manifest_path, or pass --ftm_manifest_path."
        )
    device = torch.device(
        args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
    )
    results_dir = _build_results_dir(args.results_dir, prefix=args.results_prefix)
    ftm_manifest, basis_path, core_path = _resolve_checkpoint_paths(args)
    ftm_manifest, basis_function, ind_input, _ = load_basis_and_coords(
        args.ftm_manifest_path, device
    )
    basis_path = ftm_manifest["basis_path"]
    core_path = ftm_manifest["core_path"]
    core_payload = load_core_payload(core_path, ftm_manifest=ftm_manifest)
    norm_stats = core_payload["norm_stats"]
    core_train_np = _as_core_train_array(core_payload)
    basis_function.eval()
    if hasattr(basis_function, "mode_w"):
        rank = (
            int(getattr(basis_function, "r_u")),
            int(getattr(basis_function, "r_v")),
            int(getattr(basis_function, "r_w")),
        )
    else:
        rank = (
            int(getattr(basis_function, "r_1")),
            int(getattr(basis_function, "r_2")),
            1,
        )
    data_manifest_path = args.data_manifest_path
    if ftm_manifest is not None and (not data_manifest_path):
        data_manifest_path = str(ftm_manifest.get("source_data_manifest_path", ""))
    if not data_manifest_path:
        raise ValueError(
            "Set data.data_manifest_path or use an artifact with source_data_manifest_path."
        )
    data_path, metadata_path, data_manifest = resolve_split_paths(
        data_manifest_path=data_manifest_path, split=args.split
    )
    ind_uni, data_extract, mask_np = _load_large_data_flexible(data_path, metadata_path)
    data_norm = apply_saved_normalization(data_extract, norm_stats)
    spatial_dim = 3 if int(np.asarray(ind_uni[2]).size) > 1 else 2
    data = _ensure_5d(data_norm, spatial_dim=spatial_dim)
    data, mask_np, ind_uni, time_crop_info = _crop_time_axis_to_core_if_requested(
        data, mask_np, ind_uni, core_train_np, mode=args.auto_crop_time_to_core
    )
    num_channels = int(data.shape[-1])
    shared_basis_multichannel = (
        bool(core_payload.get("shared_basis_multichannel", False)) or num_channels > 1
    )
    _check_core_compatible(
        core_train_np, data, shared_basis_multichannel, core_parameterization="per_time"
    )
    channel_names = _infer_channel_names(data_manifest, num_channels)
    mask = torch.as_tensor(np.asarray(mask_np), dtype=torch.float32)
    if mask.dim() not in (5, 6):
        raise ValueError(f"Expected 2D/3D channel-last mask, got {tuple(mask.shape)}")
    with torch.no_grad():
        basis_function.mode = "training"
        precomputed_basis = basis_function(input_ind_train=ind_input)
    pde_coord_uniques = _query_coord_uniques_for_inference(
        ind_input=ind_input,
        ind_uni=ind_uni,
        core_parameterization="per_time",
        device=device,
        spatial_axis_order=spatial_axis_order_from_manifest(
            ftm_manifest, spatial_dim=spatial_dim
        ),
    )
    num_samples = int(data.shape[0])
    batch_size = max(int(args.batch_size), 1)
    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32
    pred_batches = []
    core_batches = []
    prior_core_batches = []
    residual_core_batches = []
    metric_accum = None
    inference_batch_timings = []
    inference_sequence_seconds = []
    mean_core_np = np.mean(core_train_np, axis=0, dtype=np.float64).astype(np.float32)
    print(f"device: {device}")
    print(f"basis path: {basis_path}")
    print(f"core path: {core_path}")
    print(f"data split: {args.split}")
    print(f"test data path: {data_path}")
    print(f"results dir: {results_dir}")
    print(f"data shape: {tuple(data.shape)}")
    if time_crop_info is not None:
        print(
            "time crop:",
            f"mode={time_crop_info['mode']}",
            f"original_T={time_crop_info['original_time_length']}",
            f"cropped_T={time_crop_info['cropped_time_length']}",
            f"drop_first={time_crop_info['drop_first_frames']}",
        )
    print(
        f"shared_basis_multichannel: {shared_basis_multichannel} | num_channels: {num_channels} | channels: {channel_names}"
    )
    print(f"rank: {rank}")
    pde_dataset_type = _resolve_pde_dataset_type(
        args.pde_dataset_type,
        data_manifest_path=data_manifest_path,
        data_manifest=data_manifest,
        ftm_manifest=ftm_manifest,
    )
    pde_context = None
    pde_kwargs = (
        None
        if pde_dataset_type is None
        else _build_pde_kwargs(args, pde_dataset_type, pde_context=pde_context)
    )
    if pde_kwargs is not None and pde_dataset_type in ("ac", "ns"):
        try:
            metadata_payload = np.load(metadata_path, allow_pickle=True).item()
            metadata_data = metadata_payload.get("data", metadata_payload)
            physical_t = metadata_data.get("t_coords_physical")
            if physical_t is not None:
                physical_t = np.asarray(physical_t, dtype=np.float64).reshape(-1)
                deltas = np.diff(physical_t)
                deltas = deltas[deltas > 0.0]
                if deltas.size:
                    pde_kwargs["dt"] = float(np.median(deltas))
        except Exception:
            pass
    if pde_dataset_type == "heat3d":
        metadata_payload = (
            np.load(metadata_path, allow_pickle=True).item().get("data", {})
        )
        physical_t = metadata_payload.get("t_coords_physical")
        if physical_t is None:
            raise ValueError(
                "Heat3D inference metadata must contain t_coords_physical."
            )
        pde_kwargs["time_coords_physical"] = np.asarray(
            physical_t, dtype=np.float32
        ).reshape(-1)
    if pde_kwargs is not None:
        pde_kwargs["query_chunk_size"] = int(args.pde_query_chunk_size)
    pde_channel_names = (
        None
        if pde_dataset_type is None
        else pde_channel_order(pde_dataset_type, num_channels=num_channels)
    )
    if pde_kwargs:
        print(
            "pde evaluation:",
            f"dataset_type={pde_dataset_type}",
            f"requested={str(args.pde_dataset_type).lower()}",
            f"kwargs={pde_kwargs}",
            f"point_sample_ratio={float(args.pde_point_sample_ratio):.4f}",
            f"point_max_points={int(args.pde_point_max_points)}",
            f"query_chunk_size={int(args.pde_query_chunk_size)}",
            f"uses_pde_context={pde_context is not None}",
        )
    else:
        print("pde evaluation: disabled (pde_dataset_type=none)")
    print("prior core: train-set mean")
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = torch.arange(start, end, dtype=torch.long)
        data_batch = data[batch_idx].to(device)
        mask_batch = get_mask_batch(
            mask, batch_idx, data_batch=data_batch, device=device
        )
        _sync_device_if_needed(device)
        inference_t0 = time.perf_counter()
        with torch.no_grad():
            prior_core = _mean_prior_core_batch(
                mean_core_np, int(data_batch.shape[0]), device=device
            ).to(dtype=data_batch.dtype)
            prior_pred_norm = _decode_with_precomputed_basis(
                precomputed_basis,
                prior_core,
                shared_basis_multichannel=shared_basis_multichannel,
                core_parameterization="per_time",
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
            solved_core = prior_core + residual_core
            pred_norm = _decode_with_precomputed_basis(
                precomputed_basis,
                solved_core,
                shared_basis_multichannel=shared_basis_multichannel,
                core_parameterization="per_time",
            )
            pred_phys = inverse_normalize_tensor(pred_norm, norm_stats)
            _sync_device_if_needed(device)
            inference_elapsed = time.perf_counter() - inference_t0
            gt_phys = inverse_normalize_tensor(data_batch, norm_stats)
        batch_num = int(end - start)
        seconds_per_sequence = float(inference_elapsed / max(batch_num, 1))
        inference_batch_timings.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "batch_size": batch_num,
                "elapsed_seconds": float(inference_elapsed),
                "seconds_per_sequence": seconds_per_sequence,
            }
        )
        inference_sequence_seconds.extend([seconds_per_sequence] * batch_num)
        diff = pred_phys - gt_phys
        metric_accum = _accumulate_channel_metrics(
            metric_accum, diff, gt_phys, mask_batch
        )
        pred_np = pred_phys.detach().cpu().numpy()
        if num_channels == 1:
            pred_np = pred_np.squeeze(-1)
        pred_batches.append(pred_np.astype(save_dtype))
        core_batches.append(solved_core.detach().cpu().numpy().astype(np.float32))
        prior_core_batches.append(prior_core.detach().cpu().numpy().astype(np.float32))
        residual_core_batches.append(
            residual_core.detach().cpu().numpy().astype(np.float32)
        )
        print(
            f"[{end:3d}/{num_samples}] solved residual core batch {start:03d}:{end:03d} | prior={'mean'} | subspace={'full'} | core shape={tuple(solved_core.shape)} | inference={inference_elapsed:.6f}s ({seconds_per_sequence:.6f}s/seq)"
        )
    pred_all = np.concatenate(pred_batches, axis=0)
    core_all = np.concatenate(core_batches, axis=0)
    prior_core_all = np.concatenate(prior_core_batches, axis=0)
    residual_core_all = np.concatenate(residual_core_batches, axis=0)
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
    pde_metrics = {}
    if pde_kwargs:
        previous_query_chunk_size = _set_active_pde_query_chunk_size(
            basis_function, int(args.pde_query_chunk_size)
        )
        try:
            pde_metrics = _compute_ftm_native_query_pde_batched(
                core_np=core_all,
                data_t=data,
                norm_stats=norm_stats,
                device=device,
                basis_function=basis_function,
                shared_basis_multichannel=shared_basis_multichannel,
                coord_uniques=pde_coord_uniques,
                dataset_type=pde_dataset_type,
                pde_kwargs=pde_kwargs,
                channel_order=pde_channel_names,
                point_sample_ratio=float(args.pde_point_sample_ratio),
                point_max_points=int(args.pde_point_max_points),
                core_parameterization="per_time",
                pde_context=pde_context,
            )
        finally:
            _restore_active_pde_query_chunk_size(
                basis_function, previous_query_chunk_size
            )
    extra_metrics = {}
    if not bool(args.skip_ssim) and spatial_dim == 2:
        pred_for_extra = (
            pred_all[..., None]
            if num_channels == 1 and pred_all.ndim == 4
            else pred_all
        )
        extra_metrics = compute_ssim_metrics_batched(
            pred_np=pred_for_extra,
            data_t=data,
            norm_stats=norm_stats,
            mask_np=mask_np,
            batch_size=batch_size,
        )
    metrics_3d = {"metrics_scope": "full_volume"} if spatial_dim == 3 else {}
    inference_sequence_seconds_np = np.asarray(
        inference_sequence_seconds, dtype=np.float64
    )
    if inference_sequence_seconds_np.size > 0:
        inference_time_total = float(
            sum((item["elapsed_seconds"] for item in inference_batch_timings))
        )
        inference_time_mean = float(np.mean(inference_sequence_seconds_np))
        inference_time_median = float(np.median(inference_sequence_seconds_np))
        inference_time_min = float(np.min(inference_sequence_seconds_np))
        inference_time_max = float(np.max(inference_sequence_seconds_np))
        inference_time_p95 = float(np.percentile(inference_sequence_seconds_np, 95.0))
    else:
        inference_time_total = 0.0
        inference_time_mean = float("nan")
        inference_time_median = float("nan")
        inference_time_min = float("nan")
        inference_time_max = float("nan")
        inference_time_p95 = float("nan")
    metrics = {
        "split": args.split,
        "num_samples": num_samples,
        "rank": list(rank),
        "lambda_core_ridge": float(args.lambda_core_ridge),
        "beta_core_tv": float(args.beta_core_tv),
        "cg_max_iter": int(args.cg_max_iter),
        "cg_tol": float(args.cg_tol),
        "prior_core_mode": str("mean"),
        "auto_crop_time_to_core": str(args.auto_crop_time_to_core),
        "time_crop_info": time_crop_info,
        "pde_dataset_type": pde_dataset_type,
        "pde_kwargs": pde_kwargs,
        "pde_point_sample_ratio": float(args.pde_point_sample_ratio),
        "pde_point_max_points": int(args.pde_point_max_points),
        "pde_query_chunk_size": int(args.pde_query_chunk_size),
        "skip_ssim": bool(args.skip_ssim),
        "inference_timing_scope": "per batch: prior/residual core solve + decode + inverse normalization; excludes PDE, SSIM, plots, and file saving",
        "inference_time_total_seconds": inference_time_total,
        "inference_time_per_sequence_mean_seconds": inference_time_mean,
        "inference_time_per_sequence_median_seconds": inference_time_median,
        "inference_time_per_sequence_min_seconds": inference_time_min,
        "inference_time_per_sequence_max_seconds": inference_time_max,
        "inference_time_per_sequence_p95_seconds": inference_time_p95,
        "inference_time_per_sequence_seconds": [
            float(x) for x in inference_sequence_seconds
        ],
        "inference_batch_timings": inference_batch_timings,
        "num_channels": num_channels,
        "channel_names": channel_names,
        "mse_full": mse_full,
        "rmse": rmse_full,
        "rmse_full": rmse_full,
        "mae_full": mae_full,
        "rel_l2_full": rel_l2_full,
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
        "mask_ratio": float(np.mean(mask_np)),
        "basis_path": basis_path,
        "core_path": core_path,
        "data_manifest_path": data_manifest_path,
        "data_path": data_path,
        "metadata_path": metadata_path,
    }
    metrics.update(pde_metrics)
    metrics.update(extra_metrics)
    metrics.update(metrics_3d)
    pred_save_path = os.path.join(results_dir, "predictions_physical.npz")
    save_payload = {
        "pred": pred_all,
        "core": core_all,
        "core_prior": prior_core_all,
        "core_residual": residual_core_all,
        "t": np.asarray(ind_uni[3], dtype=np.float32),
        "u": np.asarray(ind_uni[0], dtype=np.float32),
        "v": np.asarray(ind_uni[1], dtype=np.float32),
        "channel_names": np.asarray(channel_names, dtype=object),
    }
    if spatial_dim == 3:
        save_payload["w"] = np.asarray(ind_uni[2], dtype=np.float32)
    for spectral_key in (
        "psd_radial_bin_indices",
        "psd_gt_radial",
        "psd_pred_radial",
        "psd_abs_error_radial",
        "psd_rel_error_radial",
        "psd_log_error_radial",
    ):
        if spectral_key in extra_metrics:
            dtype = np.int32 if spectral_key == "psd_radial_bin_indices" else np.float32
            save_payload[spectral_key] = np.asarray(
                extra_metrics[spectral_key], dtype=dtype
            )
    np.savez_compressed(pred_save_path, **save_payload)
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    inference_times_path = os.path.join(results_dir, "inference_times.json")
    with open(inference_times_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scope": metrics["inference_timing_scope"],
                "num_samples": num_samples,
                "batch_size": batch_size,
                "total_seconds": inference_time_total,
                "per_sequence_mean_seconds": inference_time_mean,
                "per_sequence_median_seconds": inference_time_median,
                "per_sequence_min_seconds": inference_time_min,
                "per_sequence_max_seconds": inference_time_max,
                "per_sequence_p95_seconds": inference_time_p95,
                "per_sequence_seconds": [float(x) for x in inference_sequence_seconds],
                "batch_timings": inference_batch_timings,
            },
            f,
            indent=2,
        )
    plot_count = min(max(int(args.n_plot_samples), 0), num_samples)
    time_indices = _parse_plot_time_indices(
        args.plot_time_indices, nt=pred_all.shape[1]
    )
    if plot_count > 0:
        for sample_idx in range(plot_count):
            gt_phys_np = (
                inverse_normalize_tensor(data[sample_idx : sample_idx + 1], norm_stats)
                .cpu()
                .numpy()[0]
            )
            if num_channels == 1:
                gt_phys_np = gt_phys_np.squeeze(-1)
            mask_sample = np.asarray(mask_np[sample_idx], dtype=np.float32)
            if num_channels == 1 and mask_sample.shape[-1] == 1:
                mask_sample = mask_sample[..., 0]
            plot_path = os.path.join(
                results_dir, "plots", f"sample_{sample_idx:03d}.png"
            )
            _plot_sample(
                sample_idx=sample_idx,
                pred=pred_all[sample_idx],
                gt=gt_phys_np,
                mask=mask_sample,
                time_indices=time_indices,
                out_path=plot_path,
                channel_names=channel_names,
                spatial_dim=spatial_dim,
            )
    print("reconstruction done")
    print(f"saved predictions: {pred_save_path}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved inference times: {inference_times_path}")
    print(f"saved plots dir: {os.path.join(results_dir, 'plots')}")
    print(
        "inference timing:",
        f"total={inference_time_total:.6f}s",
        f"mean_per_seq={inference_time_mean:.6f}s",
        f"median_per_seq={inference_time_median:.6f}s",
        f"p95_per_seq={inference_time_p95:.6f}s",
    )
    print(
        "full-field metrics:",
        f"mse={mse_full:.6e}",
        f"rmse={rmse_full:.6e}",
        f"mae={mae_full:.6e}",
        f"relL2={rel_l2_full:.6e}",
    )
    print(
        "sensor split metrics:",
        f"sensor_relL2={rel_l2_obs:.6e}",
        f"nonsensor_relL2={rel_l2_nonsensor:.6e}",
        f"gap={gap_nonsensor_sensor:.6e}",
    )
    if extra_metrics:
        print(
            "SSIM metrics:",
            f"ssim_mean={float(extra_metrics.get('ssim', float('nan'))):.6e}",
            f"ssim_std={float(extra_metrics.get('ssim_std', float('nan'))):.6e}",
        )
    if pde_metrics:
        print(
            "pde metrics:",
            f"pred_pde={float(pde_metrics.get('pred_pde', float('nan'))):.6e}",
            f"gt_pde={float(pde_metrics.get('gt_pde', float('nan'))):.6e}",
        )
    return RKHSInferenceResult(
        results_dir=results_dir,
        predictions_path=pred_save_path,
        metrics_path=metrics_path,
        inference_times_path=inference_times_path,
        metrics=metrics,
    )


def main(argv=None):
    args = _parse_args(argv)
    return run_rkhs_split_inference(args)


if __name__ == "__main__":
    main()
