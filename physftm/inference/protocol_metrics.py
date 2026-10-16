"""Protocol-aware reconstruction RMSE metrics."""

from __future__ import annotations
import math
import numpy as np

TEMPORAL_PROTOCOLS = {"temporal_protocol", "temporal_interval_missing"}
SPATIAL_PROTOCOLS = {"time_varying_mask", "nonuniform_mask"}
SUPPORTED_PROTOCOLS = TEMPORAL_PROTOCOLS | SPATIAL_PROTOCOLS
TEMPORAL_RMSE_KEYS = (
    "sensor_rmse_on_support_time",
    "offsensor_rmse_on_support_time",
    "sensor_rmse_on_query_time",
    "offsensor_rmse_on_query_time",
)
SPATIAL_RMSE_KEYS = ("sensor_rmse", "offsensor_rmse")


def _finite_rmse(sse, count):
    return float(math.sqrt(sse / count)) if count > 0 else float("nan")


class SquaredErrorAccumulator:

    def __init__(self, names):
        self.sse = {name: 0.0 for name in names}
        self.count = {name: 0 for name in names}

    def add(self, name, diff, mask):
        diff = np.asarray(diff, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != diff.shape[:-1]:
            raise ValueError(f"Mask {mask.shape} does not match field {diff.shape}")
        selected = diff[mask]
        if selected.size:
            self.sse[name] += float(np.sum(selected.astype(np.float64) ** 2))
            self.count[name] += int(selected.size)

    def metrics(self):
        out = {}
        for name in self.sse:
            out[name] = _finite_rmse(self.sse[name], self.count[name])
            out[f"{name}_value_count"] = int(self.count[name])
        return out


def protocol_rmse_accumulator(protocol):
    if protocol in TEMPORAL_PROTOCOLS:
        return SquaredErrorAccumulator(TEMPORAL_RMSE_KEYS)
    if protocol in SPATIAL_PROTOCOLS:
        return SquaredErrorAccumulator(SPATIAL_RMSE_KEYS)
    raise ValueError(f"Unsupported evaluation protocol {protocol!r}")


def accumulate_protocol_rmse(accumulator, *, protocol, pred, gt, sample_mask):
    diff = np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32)
    spatial_mask = np.any(np.asarray(sample_mask) > 0, axis=-1)
    if protocol in TEMPORAL_PROTOCOLS:
        support_time = np.any(spatial_mask, axis=tuple(range(1, spatial_mask.ndim)))
        sample_counts = np.sum(spatial_mask, axis=0, dtype=np.int32)
        for tid in range(int(diff.shape[0])):
            sensor = sample_counts - spatial_mask[tid].astype(np.int32) > 0
            suffix = "support_time" if bool(support_time[tid]) else "query_time"
            accumulator.add(f"sensor_rmse_on_{suffix}", diff[tid], sensor)
            accumulator.add(f"offsensor_rmse_on_{suffix}", diff[tid], ~sensor)
        return
    if protocol not in SPATIAL_PROTOCOLS:
        raise ValueError(f"Unsupported evaluation protocol {protocol!r}")
    accumulator.add("sensor_rmse", diff, spatial_mask)
    accumulator.add("offsensor_rmse", diff, ~spatial_mask)
