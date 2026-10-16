"""Spatial-axis conventions shared by training, checkpoints, and inference."""

from __future__ import annotations


def _dataset_name(dataset_type) -> str:
    value = getattr(dataset_type, "value", dataset_type)
    return str(value).strip().lower()


def spatial_axis_order_for_dataset(dataset_type, spatial_dim: int) -> tuple[str, ...]:
    """Return factor order matching the channel-last data tensor's spatial axes."""
    if int(spatial_dim) != 3:
        return ("u", "v")
    if _dataset_name(dataset_type) == "ocean":
        return ("w", "u", "v")
    return ("u", "v", "w")


def validate_spatial_axis_order(order, *, spatial_dim: int) -> tuple[str, ...]:
    expected = ("u", "v", "w") if int(spatial_dim) == 3 else ("u", "v")
    normalized = tuple(
        (
            str(axis).strip().lower()
            for axis in order
            if str(axis).strip().lower() != "t"
        )
    )
    if len(normalized) != len(expected) or set(normalized) != set(expected):
        raise ValueError(
            f"Invalid spatial axis order {normalized}; expected a permutation of {expected}."
        )
    return normalized


def spatial_axis_order_from_manifest(manifest, *, spatial_dim: int) -> tuple[str, ...]:
    """Read a saved factor order, defaulting to the legacy u/v[/w] convention."""
    if int(spatial_dim) != 3:
        return ("u", "v")
    basis = manifest.get("basis", {}) or {}
    decode = manifest.get("decode", {}) or {}
    coordinates = manifest.get("coordinates", {}) or {}
    candidates = (
        basis.get("spatial_axis_order"),
        manifest.get("spatial_axis_order"),
        decode.get("coordinate_order"),
        coordinates.get("order"),
    )
    for candidate in candidates:
        if candidate:
            return validate_spatial_axis_order(candidate, spatial_dim=3)
    return ("u", "v", "w")


def ordered_spatial_values(order, *, u, v, w=None):
    values = {"u": u, "v": v, "w": w}
    missing = [axis for axis in order if values.get(axis) is None]
    if missing:
        raise ValueError(f"Missing coordinates for spatial axes: {missing}")
    return tuple((values[axis] for axis in order))
