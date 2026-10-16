"""Normalization and normalization-cache helpers."""

import hashlib
import json
import os
import numpy as np


def _reshape_norm_stat_for_data(stat, data_shape, target_dtype=np.float32):
    arr = np.asarray(stat, dtype=target_dtype)
    if arr.ndim == 0:
        return arr
    if arr.ndim == 1 and len(data_shape) >= 1 and (arr.shape[0] == data_shape[-1]):
        return arr.reshape((1,) * (len(data_shape) - 1) + (arr.shape[0],))
    return arr


def build_norm_cache_key_payload(train_data_path, train_metadata_path, norm_mode):
    key_payload = {
        "norm_mode": str(norm_mode),
        "train_data_path": os.path.abspath(str(train_data_path)),
        "train_metadata_path": os.path.abspath(str(train_metadata_path)),
    }
    for prefix, path in (("data", train_data_path), ("meta", train_metadata_path)):
        if path and os.path.exists(path):
            key_payload[f"{prefix}_mtime_ns"] = int(os.stat(path).st_mtime_ns)
            key_payload[f"{prefix}_size"] = int(os.path.getsize(path))
    return key_payload


def build_norm_cache_path(train_data_path, train_metadata_path, norm_mode):
    cache_root = os.path.join(".", "normalization_cache")
    os.makedirs(cache_root, exist_ok=True)
    key_payload = build_norm_cache_key_payload(
        train_data_path, train_metadata_path, norm_mode
    )
    key = hashlib.md5(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(cache_root, f"{str(norm_mode).lower()}_{key}.npz")


def load_or_compute_channelwise_zscore_stats(
    data,
    train_data_path,
    train_metadata_path,
    data_manifest_path=None,
    dataset_type=None,
    data_name=None,
    eps=1e-08,
    target_dtype=np.float32,
):
    data_arr = np.asarray(data, dtype=target_dtype)
    if data_arr.ndim < 4:
        raise ValueError(
            f"channelwise_zscore expects shape (N,T,H,W) or (N,T,H,W,C), got shape={data_arr.shape}"
        )
    implicit_single_channel = data_arr.ndim == 4
    cache_path = build_norm_cache_path(
        train_data_path, train_metadata_path, norm_mode="channelwise_zscore"
    )
    cache_loaded = False
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        data_mean = np.asarray(cached["data_mean"], dtype=target_dtype)
        data_std = np.asarray(cached["data_std"], dtype=target_dtype)
        cache_compatible = data_mean.size in (
            1,
            int(data_arr.shape[-1]),
        ) and data_std.size in (1, int(data_arr.shape[-1]))
        if not cache_compatible:
            cached.close()
            data_mean = data_std = None
        elif implicit_single_channel and data_mean.size == 1:
            data_mean = np.asarray(float(data_mean.reshape(-1)[0]), dtype=target_dtype)
            data_std = np.asarray(float(data_std.reshape(-1)[0]), dtype=target_dtype)
        if cache_compatible:
            data_min = float(cached["data_min"])
            data_max = float(cached["data_max"])
            cache_loaded = True
            cached.close()
    if not cache_loaded:
        if implicit_single_channel:
            data_mean = np.asarray(
                np.mean(data_arr, dtype=np.float64), dtype=target_dtype
            )
            data_std = np.asarray(
                np.std(data_arr, dtype=np.float64), dtype=target_dtype
            )
        else:
            reduce_axes = tuple(range(data_arr.ndim - 1))
            data_mean = np.asarray(
                np.mean(data_arr, axis=reduce_axes, dtype=np.float64),
                dtype=target_dtype,
            )
            data_std = np.asarray(
                np.std(data_arr, axis=reduce_axes, dtype=np.float64), dtype=target_dtype
            )
        data_std = np.maximum(data_std, np.asarray(eps, dtype=target_dtype))
        data_min = float(np.min(data_arr))
        data_max = float(np.max(data_arr))
        key_payload = build_norm_cache_key_payload(
            train_data_path, train_metadata_path, "channelwise_zscore"
        )
        np.savez_compressed(
            cache_path,
            data_mean=data_mean,
            data_std=data_std,
            data_min=np.asarray(data_min, dtype=target_dtype),
            data_max=np.asarray(data_max, dtype=target_dtype),
            norm_mode=np.asarray("channelwise_zscore"),
            source_train_data_path=np.asarray(os.path.abspath(str(train_data_path))),
            source_train_metadata_path=np.asarray(
                os.path.abspath(str(train_metadata_path))
            ),
            source_manifest_path=np.asarray(
                ""
                if not data_manifest_path
                else os.path.abspath(str(data_manifest_path))
            ),
            source_dataset_type=np.asarray(
                "" if dataset_type is None else str(dataset_type)
            ),
            source_data_name=np.asarray("" if data_name is None else str(data_name)),
            source_cache_key_payload_json=np.asarray(
                json.dumps(key_payload, sort_keys=True)
            ),
        )
    mean_bc = _reshape_norm_stat_for_data(
        data_mean, data_arr.shape, target_dtype=target_dtype
    )
    std_bc = np.maximum(
        _reshape_norm_stat_for_data(
            data_std, data_arr.shape, target_dtype=target_dtype
        ),
        np.asarray(eps, dtype=target_dtype),
    )
    np.subtract(data_arr, mean_bc, out=data_arr)
    np.divide(data_arr, std_bc, out=data_arr)
    stats = {
        "norm_mode": "channelwise_zscore",
        "data_min": data_min,
        "data_max": data_max,
        "data_mean": data_mean,
        "data_std": data_std,
        "norm_cache_path": cache_path,
        "norm_cache_loaded": bool(cache_loaded),
        "implicit_single_channel": bool(implicit_single_channel),
    }
    return (data_arr, stats)


def apply_saved_normalization(data, norm_stats, eps=1e-08, target_dtype=np.float32):
    data_arr = np.asarray(data, dtype=target_dtype)
    mode = str(norm_stats.get("norm_mode", "none"))
    if mode == "channelwise_zscore":
        mean_bc = _reshape_norm_stat_for_data(
            norm_stats["data_mean"], data_arr.shape, target_dtype=target_dtype
        )
        std_bc = np.maximum(
            _reshape_norm_stat_for_data(
                norm_stats["data_std"], data_arr.shape, target_dtype=target_dtype
            ),
            np.asarray(eps, dtype=target_dtype),
        )
        np.subtract(data_arr, mean_bc, out=data_arr)
        np.divide(data_arr, std_bc, out=data_arr)
        normalized = data_arr
    elif mode == "channelwise_minmax":
        min_bc = _reshape_norm_stat_for_data(
            norm_stats["data_min"], data_arr.shape, target_dtype=target_dtype
        )
        range_bc = np.maximum(
            _reshape_norm_stat_for_data(
                np.asarray(norm_stats["data_max"], dtype=target_dtype)
                - np.asarray(norm_stats["data_min"], dtype=target_dtype),
                data_arr.shape,
                target_dtype=target_dtype,
            ),
            np.asarray(eps, dtype=target_dtype),
        )
        np.subtract(data_arr, min_bc, out=data_arr)
        np.divide(data_arr, range_bc, out=data_arr)
        normalized = data_arr
    elif mode == "minmax":
        np.subtract(data_arr, norm_stats["data_min"], out=data_arr)
        np.divide(
            data_arr,
            max(norm_stats["data_max"] - norm_stats["data_min"], eps),
            out=data_arr,
        )
        normalized = data_arr
    elif mode == "none":
        normalized = data_arr
    else:
        raise ValueError(f"Unsupported norm_mode: {mode}")
    return normalized
