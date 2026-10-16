import os
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physftm.artifacts.loading import inverse_normalize_tensor
from physftm.inference.io import _infer_channel_names


def _plot_sample(
    sample_idx,
    pred,
    gt,
    mask,
    time_indices,
    out_path,
    channel_names=None,
    spatial_dim=2,
):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    is_3d = int(spatial_dim) == 3
    if is_3d:
        z_center = pred.shape[1] // 2
        pred = pred[:, z_center]
        gt = gt[:, z_center]
        mask = mask[:, z_center]
    if pred.ndim == 3:
        pred = pred[..., None]
    if gt.ndim == 3:
        gt = gt[..., None]
    if mask.ndim == 3:
        mask = mask[..., None]
    if mask.shape[-1] == 1 and pred.shape[-1] != 1:
        mask = np.broadcast_to(mask, pred.shape)
    num_channels = int(pred.shape[-1])
    if channel_names is None:
        channel_names = _infer_channel_names(None, num_channels)
    nrows = len(time_indices) * num_channels
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 4.2 * nrows), squeeze=False)
    row = 0
    for tid in time_indices:
        for channel_idx in range(num_channels):
            channel_name = (
                channel_names[channel_idx]
                if channel_idx < len(channel_names)
                else f"ch{channel_idx}"
            )
            gt_t = gt[tid, :, :, channel_idx]
            pred_t = pred[tid, :, :, channel_idx]
            mask_t = mask[tid, :, :, channel_idx]
            err_t = pred_t - gt_t
            obs_t = gt_t.copy()
            obs_t[mask_t <= 0] = np.nan
            vmin = min(float(gt_t.min()), float(pred_t.min()))
            vmax = max(float(gt_t.max()), float(pred_t.max()))
            panels = [
                (gt_t, f"GT {channel_name} t={tid}", "viridis", vmin, vmax),
                (pred_t, f"Pred {channel_name} t={tid}", "viridis", vmin, vmax),
                (
                    np.abs(err_t),
                    f"|Error| {channel_name} t={tid}",
                    "magma",
                    0.0,
                    float(np.abs(err_t).max()) + 1e-12,
                ),
                (obs_t, f"Observed {channel_name} t={tid}", "viridis", vmin, vmax),
            ]
            for col, (field, title, cmap, pvmin, pvmax) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(field, origin="lower", cmap=cmap, vmin=pvmin, vmax=pvmax)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            row += 1
    fig.suptitle(f"test reconstruction sample {sample_idx:03d}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _accumulate_channel_metrics(accum, diff, gt, mask):
    num_channels = int(diff.shape[-1])
    if accum is None:
        accum = {
            "sse_full": [0.0] * num_channels,
            "sae_full": [0.0] * num_channels,
            "sgt2_full": [0.0] * num_channels,
            "count_full": [0] * num_channels,
            "sse_obs": [0.0] * num_channels,
            "sae_obs": [0.0] * num_channels,
            "sgt2_obs": [0.0] * num_channels,
            "count_obs": [0] * num_channels,
        }
    for channel_idx in range(num_channels):
        diff_c = diff[..., channel_idx]
        gt_c = gt[..., channel_idx]
        accum["sse_full"][channel_idx] += float(torch.sum(diff_c**2).item())
        accum["sae_full"][channel_idx] += float(torch.sum(torch.abs(diff_c)).item())
        accum["sgt2_full"][channel_idx] += float(torch.sum(gt_c**2).item())
        accum["count_full"][channel_idx] += int(diff_c.numel())
        mask_c = mask[..., channel_idx] > 0
        if torch.any(mask_c):
            diff_obs = diff_c[mask_c]
            gt_obs = gt_c[mask_c]
            accum["sse_obs"][channel_idx] += float(torch.sum(diff_obs**2).item())
            accum["sae_obs"][channel_idx] += float(
                torch.sum(torch.abs(diff_obs)).item()
            )
            accum["sgt2_obs"][channel_idx] += float(torch.sum(gt_obs**2).item())
            accum["count_obs"][channel_idx] += int(diff_obs.numel())
    return accum


def _finalize_channel_metrics(accum, channel_names):
    if accum is None:
        return {}
    out = {}
    for channel_idx, name in enumerate(channel_names):
        count_full = max(int(accum["count_full"][channel_idx]), 1)
        count_obs = int(accum["count_obs"][channel_idx])
        mse_full = float(accum["sse_full"][channel_idx] / count_full)
        item = {
            "mse_full": mse_full,
            "rmse_full": float(np.sqrt(mse_full)),
            "mae_full": float(accum["sae_full"][channel_idx] / count_full),
            "rel_l2_full": float(
                np.sqrt(
                    accum["sse_full"][channel_idx]
                    / max(accum["sgt2_full"][channel_idx], 1e-12)
                )
            ),
            "mse_observed": float("nan"),
            "rmse_observed": float("nan"),
            "mae_observed": float("nan"),
            "rel_l2_observed": float("nan"),
            "mse_sensor": float("nan"),
            "rmse_sensor": float("nan"),
            "mae_sensor": float("nan"),
            "rel_l2_sensor": float("nan"),
            "mse_nonsensor": float("nan"),
            "rmse_nonsensor": float("nan"),
            "mae_nonsensor": float("nan"),
            "rel_l2_nonsensor": float("nan"),
            "gap_nonsensor_sensor": float("nan"),
        }
        if count_obs > 0:
            mse_obs = float(accum["sse_obs"][channel_idx] / count_obs)
            item.update(
                {
                    "mse_observed": mse_obs,
                    "rmse_observed": float(np.sqrt(mse_obs)),
                    "mae_observed": float(accum["sae_obs"][channel_idx] / count_obs),
                    "rel_l2_observed": float(
                        np.sqrt(
                            accum["sse_obs"][channel_idx]
                            / max(accum["sgt2_obs"][channel_idx], 1e-12)
                        )
                    ),
                    "mse_sensor": mse_obs,
                    "rmse_sensor": float(np.sqrt(mse_obs)),
                    "mae_sensor": float(accum["sae_obs"][channel_idx] / count_obs),
                    "rel_l2_sensor": float(
                        np.sqrt(
                            accum["sse_obs"][channel_idx]
                            / max(accum["sgt2_obs"][channel_idx], 1e-12)
                        )
                    ),
                }
            )
        count_nonsensor = max(int(accum["count_full"][channel_idx]) - count_obs, 0)
        if count_nonsensor > 0:
            sse_nonsensor = float(
                accum["sse_full"][channel_idx] - accum["sse_obs"][channel_idx]
            )
            sae_nonsensor = float(
                accum["sae_full"][channel_idx] - accum["sae_obs"][channel_idx]
            )
            sgt2_nonsensor = float(
                accum["sgt2_full"][channel_idx] - accum["sgt2_obs"][channel_idx]
            )
            mse_nonsensor = sse_nonsensor / max(count_nonsensor, 1)
            rel_l2_nonsensor = float(
                np.sqrt(sse_nonsensor / max(sgt2_nonsensor, 1e-12))
            )
            item.update(
                {
                    "mse_nonsensor": float(mse_nonsensor),
                    "rmse_nonsensor": float(np.sqrt(mse_nonsensor)),
                    "mae_nonsensor": float(sae_nonsensor / max(count_nonsensor, 1)),
                    "rel_l2_nonsensor": rel_l2_nonsensor,
                }
            )
        rel_sensor = float(item["rel_l2_sensor"])
        rel_nonsensor = float(item["rel_l2_nonsensor"])
        if (
            np.isfinite(rel_sensor)
            and abs(rel_sensor) > 1e-12
            and np.isfinite(rel_nonsensor)
        ):
            item["gap_nonsensor_sensor"] = float(rel_nonsensor / rel_sensor)
        out[str(name)] = item
    return out


def _ensure_channel_last_5d_np(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[..., None]
    if arr.ndim != 5:
        raise ValueError(f"Expected array shape (N,T,H,W[,C]), got {arr.shape}")
    return arr


def _compute_global_ssim_data_range_from_tensor(
    data_t, norm_stats, batch_size, num_channels
):
    num_channels = int(num_channels)
    gt_min = np.full((num_channels,), np.inf, dtype=np.float64)
    gt_max = np.full((num_channels,), -np.inf, dtype=np.float64)
    num_samples = int(data_t.shape[0])
    for start in range(0, num_samples, max(int(batch_size), 1)):
        end = min(start + max(int(batch_size), 1), num_samples)
        gt_batch = (
            inverse_normalize_tensor(data_t[start:end], norm_stats)
            .detach()
            .cpu()
            .numpy()
        )
        gt_batch = _ensure_channel_last_5d_np(gt_batch)
        batch_view = gt_batch.reshape(-1, num_channels)
        gt_min = np.minimum(gt_min, np.min(batch_view, axis=0))
        gt_max = np.maximum(gt_max, np.max(batch_view, axis=0))
    if not np.all(np.isfinite(gt_min)) or not np.all(np.isfinite(gt_max)):
        raise ValueError("Unable to compute a finite global GT range for SSIM.")
    return np.maximum(gt_max - gt_min, 1e-06).astype(np.float32, copy=False)


def _prepare_ssim_data_range(ssim_data_range, num_channels):
    data_range = np.asarray(ssim_data_range, dtype=np.float32).reshape(-1)
    if data_range.size == 1 and int(num_channels) > 1:
        data_range = np.repeat(data_range, int(num_channels))
    if int(data_range.size) != int(num_channels):
        raise ValueError(
            f"Expected {int(num_channels)} SSIM data-range values, got {int(data_range.size)}"
        )
    return np.maximum(data_range, 1e-06).astype(np.float32, copy=False)


def _ssim_sample_values_from_batch(
    pred_batch_np, gt_batch_np, *, ssim_data_range, window_size=11, chunk_size=256
):
    pred_batch_np = _ensure_channel_last_5d_np(pred_batch_np)
    gt_batch_np = _ensure_channel_last_5d_np(gt_batch_np)
    num_samples, num_times, height, width, num_channels = pred_batch_np.shape
    height = int(pred_batch_np.shape[2])
    width = int(pred_batch_np.shape[3])
    window_size = max(1, min(int(window_size), height, width))
    ssim_data_range = _prepare_ssim_data_range(ssim_data_range, num_channels)
    pred_img = (
        torch.from_numpy(pred_batch_np)
        .permute(0, 1, 4, 2, 3)
        .reshape(-1, 1, height, width)
    )
    gt_img = (
        torch.from_numpy(gt_batch_np)
        .permute(0, 1, 4, 2, 3)
        .reshape(-1, 1, height, width)
    )
    num_images = int(pred_img.shape[0])
    if num_images <= 0:
        return np.zeros((0,), dtype=np.float64)
    sample_ids = np.repeat(
        np.arange(num_samples, dtype=np.int64), num_times * num_channels
    )
    channel_ids = np.tile(
        np.arange(num_channels, dtype=np.int64), num_samples * num_times
    )
    data_range_img = torch.as_tensor(
        ssim_data_range[channel_ids], dtype=torch.float32
    ).view(-1, 1, 1, 1)
    sample_ssim_sum = np.zeros((num_samples,), dtype=np.float64)
    sample_ssim_count = np.zeros((num_samples,), dtype=np.int64)
    pad = max(int(window_size) // 2, 0)
    for start in range(0, num_images, max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), num_images)
        pred_chunk = pred_img[start:end].float()
        gt_chunk = gt_img[start:end].float()
        if pad > 0:
            pred_pad = F.pad(pred_chunk, (pad, pad, pad, pad), mode="reflect")
            gt_pad = F.pad(gt_chunk, (pad, pad, pad, pad), mode="reflect")
        else:
            pred_pad = pred_chunk
            gt_pad = gt_chunk
        mu_pred = F.avg_pool2d(pred_pad, kernel_size=int(window_size), stride=1)
        mu_gt = F.avg_pool2d(gt_pad, kernel_size=int(window_size), stride=1)
        mu_pred_sq = mu_pred * mu_pred
        mu_gt_sq = mu_gt * mu_gt
        mu_pred_gt = mu_pred * mu_gt
        sigma_pred_sq = (
            F.avg_pool2d(pred_pad * pred_pad, kernel_size=int(window_size), stride=1)
            - mu_pred_sq
        )
        sigma_gt_sq = (
            F.avg_pool2d(gt_pad * gt_pad, kernel_size=int(window_size), stride=1)
            - mu_gt_sq
        )
        sigma_pred_gt = (
            F.avg_pool2d(pred_pad * gt_pad, kernel_size=int(window_size), stride=1)
            - mu_pred_gt
        )
        c1 = (0.01 * data_range_img[start:end]) ** 2
        c2 = (0.03 * data_range_img[start:end]) ** 2
        ssim_map = (
            (2.0 * mu_pred_gt + c1)
            * (2.0 * sigma_pred_gt + c2)
            / (
                (mu_pred_sq + mu_gt_sq + c1) * (sigma_pred_sq + sigma_gt_sq + c2)
                + 1e-12
            )
        )
        image_ssim = ssim_map.mean(dim=(1, 2, 3)).cpu().numpy()
        np.add.at(sample_ssim_sum, sample_ids[start:end], image_ssim)
        np.add.at(sample_ssim_count, sample_ids[start:end], 1)
    sample_valid = sample_ssim_count > 0
    sample_values = sample_ssim_sum[sample_valid] / sample_ssim_count[
        sample_valid
    ].astype(np.float64)
    return sample_values.astype(np.float64, copy=False)


def compute_ssim_metrics_batched(*, pred_np, data_t, norm_stats, mask_np, batch_size):
    pred_np = _ensure_channel_last_5d_np(pred_np)
    batch_size = max(int(batch_size), 1)
    ssim_data_range = _compute_global_ssim_data_range_from_tensor(
        data_t, norm_stats, batch_size=batch_size, num_channels=pred_np.shape[-1]
    )
    values = []
    for start in range(0, int(pred_np.shape[0]), batch_size):
        end = min(start + batch_size, int(pred_np.shape[0]))
        gt_batch = (
            inverse_normalize_tensor(data_t[start:end], norm_stats)
            .detach()
            .cpu()
            .numpy()
        )
        values.append(
            _ssim_sample_values_from_batch(
                pred_np[start:end], gt_batch, ssim_data_range=ssim_data_range
            )
        )
    ssim_values = np.concatenate(values) if values else np.zeros(0, dtype=np.float64)
    ssim_mean = float(np.mean(ssim_values)) if ssim_values.size else float("nan")
    return {
        "ssim_mean": ssim_mean,
        "ssim": ssim_mean,
        "ssim_std": float(np.std(ssim_values)) if ssim_values.size else float("nan"),
        "ssim_per_sample": ssim_values.tolist(),
    }
