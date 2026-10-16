"""Query-style finite-difference PDE losses for continuous sampled points.

The default `query_fn(coords)` callback receives continuous logical spatial
coordinates with shape `(P, 2)` and returns physical-space fields shaped
`(B, T, P)` or `(B, T, P, C)`.

For Continuous-time diagnostic queries, `query_fn(coords)` instead receives
logical `(t, x, y)` coordinates with shape `(Q, 3)` and returns `(B, Q)` or
`(B, Q, C)`.
"""

from __future__ import annotations
import math
import torch

try:
    from torch.utils.checkpoint import checkpoint as _checkpoint
except Exception:
    _checkpoint = None


def _axis_mapping_info(coord_uni, physical_length, periodic=False):
    coord = torch.as_tensor(coord_uni, dtype=torch.float32).reshape(-1)
    coord_min = float(coord.min().item())
    coord_max = float(coord.max().item())
    raw_span = max(coord_max - coord_min, 1e-08)
    if coord.numel() > 1:
        diffs = torch.diff(coord)
        positive = diffs[diffs > 0]
        step = (
            float(torch.median(positive).item())
            if positive.numel() > 0
            else raw_span / max(int(coord.numel()) - 1, 1)
        )
    else:
        step = raw_span
    logical_span = step * (
        int(coord.numel()) if periodic else max(int(coord.numel()) - 1, 1)
    )
    logical_span = max(float(logical_span), 1e-08)
    physical_length = max(float(physical_length), 1e-08)
    use_identity = abs(logical_span - physical_length) / max(
        abs(physical_length), 1e-08
    ) <= 0.05 and abs(coord_min) <= 0.05 * max(abs(physical_length), 1.0)
    return {
        "coord_min": coord_min,
        "coord_max": coord_max,
        "logical_span": logical_span,
        "physical_length": physical_length,
        "identity": bool(use_identity),
        "nominal_step_physical": physical_length
        / max(int(coord.numel()) if periodic else max(int(coord.numel()) - 1, 1), 1),
    }


def _physical_to_logical_delta(physical_delta, coord_uni, physical_length, periodic):
    info = _axis_mapping_info(
        coord_uni, physical_length=physical_length, periodic=periodic
    )
    if physical_delta is None or float(physical_delta) <= 0.0:
        physical_delta = info["nominal_step_physical"]
    logical_delta = float(physical_delta) * (
        info["logical_span"] / info["physical_length"]
    )
    return (float(physical_delta), float(logical_delta), info)


def _logical_to_physical(values, info):
    if torch.is_tensor(values):
        values = values.to(dtype=torch.float32)
    else:
        values = torch.as_tensor(values, dtype=torch.float32)
    if info["identity"] or info["physical_length"] <= 0.0:
        return values
    return (values - float(info["coord_min"])) * (
        float(info["physical_length"]) / float(info["logical_span"])
    )


def _periodic_shift_axis(values, delta, info):
    coord_min = float(info["coord_min"])
    logical_span = float(info["logical_span"])
    return torch.remainder(values - coord_min + float(delta), logical_span) + coord_min


def _temporal_central_diff(field, dt):
    if int(field.shape[1]) < 3:
        raise ValueError("Query FD PDE residual requires at least 3 time steps.")
    return (field[:, 2:] - field[:, :-2]) / (2.0 * float(dt))


def _validate_fd_order(order, *, name, supported=(2, 4)):
    order = int(order)
    if order not in supported:
        raise ValueError(f"Unsupported {name}={order}. Supported values: {supported}")
    return order


def _center_time_slice_ordered(field, order):
    order = _validate_fd_order(order, name="fd_time_order")
    trim = order // 2
    if int(field.shape[1]) < 2 * trim + 1:
        raise ValueError(
            f"Query FD PDE residual with fd_time_order={order} requires at least {2 * trim + 1} time steps."
        )
    return field[:, trim:-trim]


def _temporal_central_diff_ordered(field, dt, order):
    order = _validate_fd_order(order, name="fd_time_order")
    if order == 2:
        return _temporal_central_diff(field, dt)
    if int(field.shape[1]) < 5:
        raise ValueError(
            "Query FD PDE residual with fd_time_order=4 requires at least 5 time steps."
        )
    return (
        -field[:, 4:] + 8.0 * field[:, 3:-1] - 8.0 * field[:, 1:-3] + field[:, :-4]
    ) / (12.0 * float(dt))


def _fd_first_staggered(value_fn, *, axis, step, order, center_u=0.0, center_v=0.0):
    order = _validate_fd_order(order, name="fd_space_order")
    if int(axis) == 0:
        eval_fn = lambda offset: value_fn(center_u + float(offset), center_v)
    elif int(axis) == 1:
        eval_fn = lambda offset: value_fn(center_u, center_v + float(offset))
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    if order == 2:
        return (eval_fn(+0.5) - eval_fn(-0.5)) / float(step)
    return (
        1.0 / 24.0 * eval_fn(-1.5)
        - 9.0 / 8.0 * eval_fn(-0.5)
        + 9.0 / 8.0 * eval_fn(+0.5)
        - 1.0 / 24.0 * eval_fn(+1.5)
    ) / float(step)


def _fd_second_centered(value_fn, *, axis, step, order, center_u=0.0, center_v=0.0):
    order = _validate_fd_order(order, name="fd_space_order")
    if int(axis) == 0:
        eval_fn = lambda offset: value_fn(center_u + float(offset), center_v)
    elif int(axis) == 1:
        eval_fn = lambda offset: value_fn(center_u, center_v + float(offset))
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    center = eval_fn(0.0)
    if order == 2:
        return (eval_fn(+1.0) - 2.0 * center + eval_fn(-1.0)) / float(step) ** 2
    return (
        -eval_fn(+2.0)
        + 16.0 * eval_fn(+1.0)
        - 30.0 * center
        + 16.0 * eval_fn(-1.0)
        - eval_fn(-2.0)
    ) / (12.0 * float(step) ** 2)


def _normalize_query_output(field):
    if field.dim() == 3:
        return field.unsqueeze(-1)
    if field.dim() == 4:
        return field
    raise ValueError(
        f"query_fn must return (B,T,P) or (B,T,P,C), got {tuple(field.shape)}"
    )


def _normalize_time_query_output(field):
    if field.dim() == 2:
        return field.unsqueeze(-1)
    if field.dim() == 3:
        return field
    raise ValueError(
        f"time-query query_fn must return (B,Q) or (B,Q,C), got {tuple(field.shape)}"
    )


def build_dense_field_query_fn_2d(field, coord_u, coord_v, *, periodic_u, periodic_v):
    """Build a differentiable continuous query function from dense 2D fields.

    Args:
      field:
        `(B,T,H,W)` or `(B,T,H,W,C)` physical-space dense field.
      coord_u / coord_v:
        1D logical coordinates for the dense grid along the two spatial axes.
      periodic_u / periodic_v:
        Whether the corresponding axis wraps periodically.

    Returns:
      query_fn(query_coords) -> `(B,T,P)` or `(B,T,P,C)` matching the input
      channel convention.
    """
    field_t = torch.as_tensor(field)
    squeeze_channel = False
    if field_t.dim() == 4:
        field_t = field_t.unsqueeze(-1)
        squeeze_channel = True
    if field_t.dim() != 5:
        raise ValueError(
            f"field must have shape (B,T,H,W) or (B,T,H,W,C), got {tuple(field_t.shape)}"
        )
    coord_u_t = torch.as_tensor(
        coord_u, dtype=field_t.dtype, device=field_t.device
    ).reshape(-1)
    coord_v_t = torch.as_tensor(
        coord_v, dtype=field_t.dtype, device=field_t.device
    ).reshape(-1)
    h = int(coord_u_t.numel())
    w = int(coord_v_t.numel())
    if h <= 0 or w <= 0:
        raise ValueError("coord_u and coord_v must be non-empty.")
    if int(field_t.shape[2]) != h or int(field_t.shape[3]) != w:
        raise ValueError(
            f"Dense field spatial shape {(int(field_t.shape[2]), int(field_t.shape[3]))} does not match coord lengths {(h, w)}"
        )

    def _fractional_index(values, coord, periodic):
        values = torch.as_tensor(values, dtype=field_t.dtype, device=field_t.device)
        if int(coord.numel()) == 1:
            return torch.zeros_like(values)
        diffs = torch.diff(coord)
        positive = diffs[diffs > 0]
        if positive.numel() > 0:
            step = float(torch.median(positive).item())
        else:
            step = float((coord[-1] - coord[0]).item()) / max(int(coord.numel()) - 1, 1)
        step = max(step, 1e-08)
        base = float(coord[0].item())
        idx = (values - base) / step
        if bool(periodic):
            return torch.remainder(idx, float(coord.numel()))
        return torch.clamp(idx, min=0.0, max=max(float(coord.numel() - 1), 0.0))

    flat = field_t.reshape(
        int(field_t.shape[0]), int(field_t.shape[1]), h * w, int(field_t.shape[-1])
    )

    def query_fn(query_coords):
        query_coords_t = torch.as_tensor(
            query_coords, dtype=field_t.dtype, device=field_t.device
        )
        if query_coords_t.dim() != 2 or int(query_coords_t.shape[-1]) != 2:
            raise ValueError(
                f"query_coords must have shape (P,2), got {tuple(query_coords_t.shape)}"
            )
        u_idx_f = _fractional_index(query_coords_t[:, 0], coord_u_t, periodic_u)
        v_idx_f = _fractional_index(query_coords_t[:, 1], coord_v_t, periodic_v)
        u0 = torch.floor(u_idx_f).to(dtype=torch.long)
        v0 = torch.floor(v_idx_f).to(dtype=torch.long)
        if bool(periodic_u):
            u1 = torch.remainder(u0 + 1, h)
        else:
            u1 = torch.clamp(u0 + 1, max=h - 1)
        if bool(periodic_v):
            v1 = torch.remainder(v0 + 1, w)
        else:
            v1 = torch.clamp(v0 + 1, max=w - 1)
        wu = (u_idx_f - u0.to(dtype=field_t.dtype)).reshape(1, 1, -1, 1)
        wv = (v_idx_f - v0.to(dtype=field_t.dtype)).reshape(1, 1, -1, 1)

        def _gather(ui, vj):
            idx = (ui * w + vj).to(dtype=torch.long)
            return flat.index_select(dim=2, index=idx)

        f00 = _gather(u0, v0)
        f01 = _gather(u0, v1)
        f10 = _gather(u1, v0)
        f11 = _gather(u1, v1)
        sampled = (
            (1.0 - wu) * (1.0 - wv) * f00
            + (1.0 - wu) * wv * f01
            + wu * (1.0 - wv) * f10
            + wu * wv * f11
        )
        if squeeze_channel:
            return sampled[..., 0]
        return sampled

    return query_fn


def _extract_named_channel(field, channel_order, name):
    if field.dim() not in (3, 4):
        raise ValueError(
            f"Expected field shape (B,T,P,C) or paired (B,Q,C), got {tuple(field.shape)}"
        )
    channel_order = tuple((str(x) for x in channel_order))
    if name not in channel_order:
        raise ValueError(f"Channel {name!r} not found in channel_order={channel_order}")
    return field[..., channel_order.index(name)]


def _mse(field):
    return torch.mean(field**2)


def _mse_sum(*fields):
    total = None
    for field in fields:
        value = _mse(field)
        total = value if total is None else total + value
    return total


def _build_ns_forcing_pointwise(x_phys, y_phys, *, Lx, Ly, scale):
    phase = (
        2.0
        * math.pi
        * (x_phys / max(float(Lx), 1e-08) + y_phys / max(float(Ly), 1e-08))
    )
    return float(scale) * (torch.sin(phase) + torch.cos(phase))


class _FDQueryField2D:
    """Manage continuous query points, physical step sizes, and boundary handling."""

    def __init__(
        self,
        *,
        query_fn,
        sample_coords,
        coord_u,
        coord_v,
        physical_length_u,
        physical_length_v,
        periodic_u,
        periodic_v,
        fd_du=None,
        fd_dv=None,
        max_offset_u=2,
        max_offset_v=2,
    ):
        sample_coords = (
            sample_coords.to(dtype=torch.float32)
            if torch.is_tensor(sample_coords)
            else torch.as_tensor(sample_coords, dtype=torch.float32)
        )
        if sample_coords.dim() != 2 or int(sample_coords.shape[-1]) not in (2, 3):
            raise ValueError(
                f"sample_coords must have shape (P,2) or paired (Q,3), got {tuple(sample_coords.shape)}"
            )
        self.paired_time_space = int(sample_coords.shape[-1]) == 3
        self.query_fn = query_fn
        self.du_phys, self.du_logical, self.info_u = _physical_to_logical_delta(
            fd_du, coord_u, physical_length_u, periodic_u
        )
        self.dv_phys, self.dv_logical, self.info_v = _physical_to_logical_delta(
            fd_dv, coord_v, physical_length_v, periodic_v
        )
        self.periodic_u = bool(periodic_u)
        self.periodic_v = bool(periodic_v)
        self.cache = {}
        base = (
            sample_coords[:, 1:].clone()
            if self.paired_time_space
            else sample_coords.clone()
        )
        if not self.periodic_u:
            base[:, 0] = self._clamp_interior(
                base[:, 0], self.info_u, self.du_logical, max_offset_u
            )
        if not self.periodic_v:
            base[:, 1] = self._clamp_interior(
                base[:, 1], self.info_v, self.dv_logical, max_offset_v
            )
        self.base_coords = base

    @staticmethod
    def _clamp_interior(values, info, logical_delta, max_offset):
        margin = (max(int(max_offset), 0) + 0.001) * float(logical_delta)
        lo = float(info["coord_min"]) + margin
        hi = float(info["coord_max"]) - margin
        if hi <= lo:
            return torch.full_like(
                values, 0.5 * (float(info["coord_min"]) + float(info["coord_max"]))
            )
        return torch.clamp(values, min=lo, max=hi)

    def _shift_axis(self, values, offset, axis):
        if axis == 0:
            delta = float(offset) * self.du_logical
            info = self.info_u
            periodic = self.periodic_u
        else:
            delta = float(offset) * self.dv_logical
            info = self.info_v
            periodic = self.periodic_v
        if periodic:
            return _periodic_shift_axis(values, delta, info)
        return torch.clamp(
            values + delta, min=float(info["coord_min"]), max=float(info["coord_max"])
        )

    def shifted_coords(self, offset_u=0.0, offset_v=0.0):
        coords = self.base_coords.clone()
        if float(offset_u) != 0.0:
            coords[:, 0] = self._shift_axis(coords[:, 0], float(offset_u), axis=0)
        if float(offset_v) != 0.0:
            coords[:, 1] = self._shift_axis(coords[:, 1], float(offset_v), axis=1)
        return coords

    def query(self, offset_u=0.0, offset_v=0.0):
        key = (float(offset_u), float(offset_v))
        should_cache = not torch.is_grad_enabled()
        if should_cache and key in self.cache:
            return self.cache[key]
        coords = self.shifted_coords(offset_u, offset_v)
        queried = _normalize_query_output(self.query_fn(coords))
        if should_cache:
            self.cache[key] = queried
            return self.cache[key]
        return queried


class _FDQueryFieldTimeSpace2D:
    """Continuous `(t, x, y)` query sampler for continuous-time PDE queries."""

    def __init__(
        self,
        *,
        query_fn,
        sample_coords,
        sample_t_coords,
        coord_t,
        coord_u,
        coord_v,
        dt,
        physical_length_u,
        physical_length_v,
        periodic_u,
        periodic_v,
        fd_du=None,
        fd_dv=None,
        max_offset_u=2,
        max_offset_v=2,
        time_order=2,
    ):
        sample_coords = (
            sample_coords.to(dtype=torch.float32)
            if torch.is_tensor(sample_coords)
            else torch.as_tensor(sample_coords, dtype=torch.float32)
        )
        if sample_coords.dim() != 2 or int(sample_coords.shape[-1]) not in (2, 3):
            raise ValueError(
                f"sample_coords must have shape (P,2) or paired (Q,3), got {tuple(sample_coords.shape)}"
            )
        self.paired_time_space = int(sample_coords.shape[-1]) == 3
        coord_t_t = torch.as_tensor(coord_t, dtype=torch.float32).reshape(-1)
        trim = _validate_fd_order(time_order, name="fd_time_order") // 2
        if int(coord_t_t.numel()) < 2 * trim + 1:
            raise ValueError(
                f"Continuous-time query PDE residual with fd_time_order={time_order} requires at least {2 * trim + 1} time steps."
            )
        self.query_fn = query_fn
        self.periodic_u = bool(periodic_u)
        self.periodic_v = bool(periodic_v)
        self.cache = {}
        self.dt_phys, self.dt_logical, self.info_t = _physical_to_logical_delta(
            dt,
            coord_t_t,
            max(float(dt) * max(int(coord_t_t.numel()) - 1, 1), 1e-08),
            False,
        )
        self.du_phys, self.du_logical, self.info_u = _physical_to_logical_delta(
            fd_du, coord_u, physical_length_u, periodic_u
        )
        self.dv_phys, self.dv_logical, self.info_v = _physical_to_logical_delta(
            fd_dv, coord_v, physical_length_v, periodic_v
        )
        base = (
            sample_coords[:, 1:].clone()
            if self.paired_time_space
            else sample_coords.clone()
        )
        if not self.periodic_u:
            base[:, 0] = _FDQueryField2D._clamp_interior(
                base[:, 0], self.info_u, self.du_logical, max_offset_u
            )
        if not self.periodic_v:
            base[:, 1] = _FDQueryField2D._clamp_interior(
                base[:, 1], self.info_v, self.dv_logical, max_offset_v
            )
        self.base_coords = base
        if self.paired_time_space:
            base_times = sample_coords[:, 0].to(dtype=torch.float32).reshape(-1)
            base_times = _FDQueryField2D._clamp_interior(
                base_times, self.info_t, self.dt_logical, trim
            )
        elif sample_t_coords is None:
            base_times = coord_t_t[trim:-trim].clone()
        else:
            base_times = (
                sample_t_coords.to(dtype=torch.float32)
                if torch.is_tensor(sample_t_coords)
                else torch.as_tensor(sample_t_coords, dtype=torch.float32)
            ).reshape(-1)
            if int(base_times.numel()) <= 0:
                raise ValueError("sample_t_coords must be non-empty when provided.")
            base_times = _FDQueryField2D._clamp_interior(
                base_times, self.info_t, self.dt_logical, trim
            )
        self.base_times = base_times
        self.num_times = int(self.base_times.numel())
        self.num_points = int(self.base_coords.shape[0])

    def _shift_space_axis(self, values, offset, axis):
        if axis == 0:
            delta = float(offset) * self.du_logical
            info = self.info_u
            periodic = self.periodic_u
        else:
            delta = float(offset) * self.dv_logical
            info = self.info_v
            periodic = self.periodic_v
        if periodic:
            return _periodic_shift_axis(values, delta, info)
        return torch.clamp(
            values + delta, min=float(info["coord_min"]), max=float(info["coord_max"])
        )

    def query(self, offset_t=0.0, offset_u=0.0, offset_v=0.0):
        key = (float(offset_t), float(offset_u), float(offset_v))
        should_cache = not torch.is_grad_enabled()
        if should_cache and key in self.cache:
            return self.cache[key]
        if self.paired_time_space:
            t_grid = self.base_times + float(offset_t) * self.dt_logical
            u_grid = self.base_coords[:, 0].clone()
            v_grid = self.base_coords[:, 1].clone()
        else:
            times = self.base_times + float(offset_t) * self.dt_logical
            t_grid = times[:, None].expand(self.num_times, self.num_points).reshape(-1)
            u_grid = (
                self.base_coords[:, 0][None, :]
                .expand(self.num_times, self.num_points)
                .reshape(-1)
            )
            v_grid = (
                self.base_coords[:, 1][None, :]
                .expand(self.num_times, self.num_points)
                .reshape(-1)
            )
        if float(offset_u) != 0.0:
            u_grid = self._shift_space_axis(u_grid, float(offset_u), axis=0)
        if float(offset_v) != 0.0:
            v_grid = self._shift_space_axis(v_grid, float(offset_v), axis=1)
        coords = torch.stack([t_grid, u_grid, v_grid], dim=-1)
        queried = _normalize_time_query_output(self.query_fn(coords))
        if self.paired_time_space:
            cached = queried
        else:
            bsz = int(queried.shape[0])
            num_channels = int(queried.shape[-1])
            cached = queried.reshape(bsz, self.num_times, self.num_points, num_channels)
        if should_cache:
            self.cache[key] = cached
            return self.cache[key]
        return cached


def _temporal_first_diff_query(eval_fn, dt, order):
    order = _validate_fd_order(order, name="fd_time_order")
    if order == 2:
        return (eval_fn(+1.0) - eval_fn(-1.0)) / (2.0 * float(dt))
    return (
        -eval_fn(+2.0) + 8.0 * eval_fn(+1.0) - 8.0 * eval_fn(-1.0) + eval_fn(-2.0)
    ) / (12.0 * float(dt))


def _compute_ac_fd_query_terms(
    *,
    query_fn,
    sample_coords,
    sample_t_coords=None,
    coord_u,
    coord_v,
    dt,
    Lx,
    Ly,
    ac_lambda,
    epsilon,
    fd_dx=None,
    fd_dy=None,
    fd_space_order=2,
    fd_time_order=2,
    coord_t=None,
    query_time_continuous=False,
):
    fd_space_order = _validate_fd_order(fd_space_order, name="fd_space_order")
    fd_time_order = _validate_fd_order(fd_time_order, name="fd_time_order")
    max_offset = 1 if fd_space_order == 2 else 2
    if query_time_continuous:
        if coord_t is None:
            raise ValueError("coord_t is required when query_time_continuous=True")
        sampler = _FDQueryFieldTimeSpace2D(
            query_fn=query_fn,
            sample_coords=sample_coords,
            sample_t_coords=sample_t_coords,
            coord_t=coord_t,
            coord_u=coord_u,
            coord_v=coord_v,
            dt=dt,
            physical_length_u=Lx,
            physical_length_v=Ly,
            periodic_u=True,
            periodic_v=True,
            fd_du=fd_dx,
            fd_dv=fd_dy,
            max_offset_u=max_offset,
            max_offset_v=max_offset,
            time_order=fd_time_order,
        )
        dx = float(sampler.du_phys)
        dy = float(sampler.dv_phys)
        U = lambda ot, ox, oy: sampler.query(ot, ox, oy)[..., 0]
        u_c = U(0.0, 0.0, 0.0)
        u_value = lambda ox, oy: U(0.0, ox, oy)
        u_xx = _fd_second_centered(u_value, axis=0, step=dx, order=fd_space_order)
        u_yy = _fd_second_centered(u_value, axis=1, step=dy, order=fd_space_order)
        u_t = _temporal_first_diff_query(lambda ot: U(ot, 0.0, 0.0), dt, fd_time_order)
        lap = u_xx + u_yy
    else:
        sampler = _FDQueryField2D(
            query_fn=query_fn,
            sample_coords=sample_coords,
            coord_u=coord_u,
            coord_v=coord_v,
            physical_length_u=Lx,
            physical_length_v=Ly,
            periodic_u=True,
            periodic_v=True,
            fd_du=fd_dx,
            fd_dv=fd_dy,
            max_offset_u=max_offset,
            max_offset_v=max_offset,
        )
        u = sampler.query(0.0, 0.0)[..., 0]
        dx = float(sampler.du_phys)
        dy = float(sampler.dv_phys)
        if fd_space_order == 2:
            u_xx = (
                sampler.query(+1.0, 0.0)[..., 0]
                - 2.0 * u
                + sampler.query(-1.0, 0.0)[..., 0]
            ) / dx**2
            u_yy = (
                sampler.query(0.0, +1.0)[..., 0]
                - 2.0 * u
                + sampler.query(0.0, -1.0)[..., 0]
            ) / dy**2
        else:
            u_xx = (
                -sampler.query(+2.0, 0.0)[..., 0]
                + 16.0 * sampler.query(+1.0, 0.0)[..., 0]
                - 30.0 * u
                + 16.0 * sampler.query(-1.0, 0.0)[..., 0]
                - sampler.query(-2.0, 0.0)[..., 0]
            ) / (12.0 * dx**2)
            u_yy = (
                -sampler.query(0.0, +2.0)[..., 0]
                + 16.0 * sampler.query(0.0, +1.0)[..., 0]
                - 30.0 * u
                + 16.0 * sampler.query(0.0, -1.0)[..., 0]
                - sampler.query(0.0, -2.0)[..., 0]
            ) / (12.0 * dy**2)
        u_t = _temporal_central_diff_ordered(u, dt, fd_time_order)
        u_c = _center_time_slice_ordered(u, fd_time_order)
        lap = _center_time_slice_ordered(u_xx + u_yy, fd_time_order)
    diffusion = float(ac_lambda) * float(epsilon) ** 2 * lap
    nonlinear = float(ac_lambda) * (u_c**3 - u_c)
    residual = u_t - diffusion + nonlinear
    ref_field = u_c
    return {
        "residual": residual,
        "u_t": u_t,
        "u_c": u_c,
        "lap": lap,
        "diffusion": diffusion,
        "nonlinear": nonlinear,
        "fd_dx": torch.as_tensor(dx, dtype=ref_field.dtype, device=ref_field.device),
        "fd_dy": torch.as_tensor(dy, dtype=ref_field.dtype, device=ref_field.device),
    }


def compute_ac_pde_loss_fd_query(
    *,
    query_fn,
    sample_coords,
    sample_t_coords=None,
    coord_u,
    coord_v,
    dt,
    Lx,
    Ly,
    ac_lambda,
    epsilon,
    fd_dx=None,
    fd_dy=None,
    fd_space_order=2,
    fd_time_order=2,
    coord_t=None,
    query_time_continuous=False,
    eps=1e-08,
):
    del eps
    terms = _compute_ac_fd_query_terms(
        query_fn=query_fn,
        sample_coords=sample_coords,
        sample_t_coords=sample_t_coords,
        coord_u=coord_u,
        coord_v=coord_v,
        dt=dt,
        Lx=Lx,
        Ly=Ly,
        ac_lambda=ac_lambda,
        epsilon=epsilon,
        fd_dx=fd_dx,
        fd_dy=fd_dy,
        fd_space_order=fd_space_order,
        fd_time_order=fd_time_order,
        coord_t=coord_t,
        query_time_continuous=query_time_continuous,
    )
    residual = terms["residual"]
    u_t = terms["u_t"]
    diffusion = terms["diffusion"]
    lap = terms["lap"]
    return {
        "pde": _mse(residual),
        "term_residual": _mse(residual),
        "term_u_t": _mse(u_t),
        "term_diffusion": _mse(diffusion),
        "term_lap": _mse(lap),
        "fd_dx": terms["fd_dx"],
        "fd_dy": terms["fd_dy"],
    }


def compute_ns_pde_loss_fd_query(
    *,
    query_fn,
    sample_coords,
    sample_t_coords=None,
    coord_u,
    coord_v,
    dt,
    Lx,
    Ly,
    nu,
    forcing_scale,
    channel_order,
    fd_dx=None,
    fd_dy=None,
    fd_space_order=2,
    fd_time_order=2,
    coord_t=None,
    query_time_continuous=False,
    return_terms=False,
):
    """2D Navier-Stokes fd-query residual on periodic `(u, v, omega)` fields.

    Query loss uses the multichannel local form
        ∂_t ω + u ω_x + v ω_y = ν (ω_xx + ω_yy) + f,
        div(u) = u_x + v_y = 0,
        ω = v_x - u_y.

    This avoids the dense global Poisson inversion `-Δψ = ω` required by the
    omega-only representation. Therefore the query loss requires channels
    `("u", "v", "omega")`.
    """
    fd_space_order = _validate_fd_order(fd_space_order, name="fd_space_order")
    fd_time_order = _validate_fd_order(fd_time_order, name="fd_time_order")
    required_channels = ("u", "v", "omega")
    channel_names = tuple((str(x) for x in channel_order))
    missing = [name for name in required_channels if name not in channel_names]
    if missing:
        raise ValueError(
            f"NS fd-query PDE loss requires channel_order containing ('u', 'v', 'omega'); missing={missing}, got={channel_names}"
        )
    max_offset = 1 if fd_space_order == 2 else 2
    if query_time_continuous:
        if coord_t is None:
            raise ValueError("coord_t is required when query_time_continuous=True")
        sampler = _FDQueryFieldTimeSpace2D(
            query_fn=query_fn,
            sample_coords=sample_coords,
            sample_t_coords=sample_t_coords,
            coord_t=coord_t,
            coord_u=coord_u,
            coord_v=coord_v,
            dt=dt,
            physical_length_u=Lx,
            physical_length_v=Ly,
            periodic_u=True,
            periodic_v=True,
            fd_du=fd_dx,
            fd_dv=fd_dy,
            max_offset_u=max_offset,
            max_offset_v=max_offset,
            time_order=fd_time_order,
        )
        dx = float(sampler.du_phys)
        dy = float(sampler.dv_phys)

        def Q(name, ot=0.0, ox=0.0, oy=0.0):
            return _extract_named_channel(
                sampler.query(ot, ox, oy), channel_names, name
            )

        u = Q("u", 0.0, 0.0, 0.0)
        v = Q("v", 0.0, 0.0, 0.0)
        omega = Q("omega", 0.0, 0.0, 0.0)
        u_value = lambda ox, oy: Q("u", 0.0, ox, oy)
        v_value = lambda ox, oy: Q("v", 0.0, ox, oy)
        omega_value = lambda ox, oy: Q("omega", 0.0, ox, oy)
    else:
        sampler = _FDQueryField2D(
            query_fn=query_fn,
            sample_coords=sample_coords,
            coord_u=coord_u,
            coord_v=coord_v,
            physical_length_u=Lx,
            physical_length_v=Ly,
            periodic_u=True,
            periodic_v=True,
            fd_du=fd_dx,
            fd_dv=fd_dy,
            max_offset_u=max_offset,
            max_offset_v=max_offset,
        )
        dx = float(sampler.du_phys)
        dy = float(sampler.dv_phys)

        def Q(name, ox=0.0, oy=0.0):
            return _extract_named_channel(sampler.query(ox, oy), channel_names, name)

        u = Q("u", 0.0, 0.0)
        v = Q("v", 0.0, 0.0)
        omega = Q("omega", 0.0, 0.0)
        u_value = lambda ox, oy: Q("u", ox, oy)
        v_value = lambda ox, oy: Q("v", ox, oy)
        omega_value = lambda ox, oy: Q("omega", ox, oy)
    u_x = _fd_first_staggered(u_value, axis=0, step=dx, order=fd_space_order)
    u_y = _fd_first_staggered(u_value, axis=1, step=dy, order=fd_space_order)
    v_x = _fd_first_staggered(v_value, axis=0, step=dx, order=fd_space_order)
    v_y = _fd_first_staggered(v_value, axis=1, step=dy, order=fd_space_order)
    omega_x = _fd_first_staggered(omega_value, axis=0, step=dx, order=fd_space_order)
    omega_y = _fd_first_staggered(omega_value, axis=1, step=dy, order=fd_space_order)
    omega_xx = _fd_second_centered(omega_value, axis=0, step=dx, order=fd_space_order)
    omega_yy = _fd_second_centered(omega_value, axis=1, step=dy, order=fd_space_order)
    point_shape = (
        (1, -1)
        if query_time_continuous and getattr(sampler, "paired_time_space", False)
        else (1, 1, -1)
    )
    x_phys = (
        _logical_to_physical(sampler.base_coords[:, 0], sampler.info_u)
        .reshape(point_shape)
        .to(dtype=u.dtype, device=u.device)
    )
    y_phys = (
        _logical_to_physical(sampler.base_coords[:, 1], sampler.info_v)
        .reshape(point_shape)
        .to(dtype=u.dtype, device=u.device)
    )
    forcing = _build_ns_forcing_pointwise(
        x_phys, y_phys, Lx=Lx, Ly=Ly, scale=forcing_scale
    )
    if query_time_continuous:
        u_c = u
        v_c = v
        omega_c = omega
        u_x_c = u_x
        u_y_c = u_y
        v_x_c = v_x
        v_y_c = v_y
        omega_x_c = omega_x
        omega_y_c = omega_y
        laplace_omega_c = omega_xx + omega_yy
        omega_t = _temporal_first_diff_query(
            lambda ot: Q("omega", ot, 0.0, 0.0), dt, fd_time_order
        )
    else:
        u_c = _center_time_slice_ordered(u, fd_time_order)
        v_c = _center_time_slice_ordered(v, fd_time_order)
        omega_c = _center_time_slice_ordered(omega, fd_time_order)
        u_x_c = _center_time_slice_ordered(u_x, fd_time_order)
        u_y_c = _center_time_slice_ordered(u_y, fd_time_order)
        v_x_c = _center_time_slice_ordered(v_x, fd_time_order)
        v_y_c = _center_time_slice_ordered(v_y, fd_time_order)
        omega_x_c = _center_time_slice_ordered(omega_x, fd_time_order)
        omega_y_c = _center_time_slice_ordered(omega_y, fd_time_order)
        laplace_omega_c = _center_time_slice_ordered(omega_xx + omega_yy, fd_time_order)
        omega_t = _temporal_central_diff_ordered(omega, dt, fd_time_order)
    forcing_c = forcing.expand_as(omega_t)
    adv_x = u_c * omega_x_c
    adv_y = v_c * omega_y_c
    diffusion = -float(nu) * laplace_omega_c
    vorticity_residual = omega_t + adv_x + adv_y + diffusion - forcing_c
    incompressibility_residual = u_x_c + v_y_c
    vorticity_definition_residual = omega_c - (v_x_c - u_y_c)
    if return_terms:
        return {
            "residual_sq": vorticity_residual**2
            + incompressibility_residual**2
            + vorticity_definition_residual**2,
            "vorticity_residual": vorticity_residual,
            "incompressibility_residual": incompressibility_residual,
            "vorticity_definition_residual": vorticity_definition_residual,
            "omega_t": omega_t,
            "adv_x": adv_x,
            "adv_y": adv_y,
            "diffusion": diffusion,
            "forcing": forcing_c,
            "u_x": u_x_c,
            "v_y": v_y_c,
            "v_x": v_x_c,
            "u_y": u_y_c,
            "fd_dx": torch.as_tensor(sampler.du_phys, dtype=u.dtype, device=u.device),
            "fd_dy": torch.as_tensor(sampler.dv_phys, dtype=u.dtype, device=u.device),
        }
    return {
        "pde": _mse_sum(
            vorticity_residual,
            incompressibility_residual,
            vorticity_definition_residual,
        ),
        "term_vorticity_residual": _mse(vorticity_residual),
        "term_omega_t": _mse(omega_t),
        "term_adv_x": _mse(adv_x),
        "term_adv_y": _mse(adv_y),
        "term_diffusion": _mse(diffusion),
        "term_forcing": _mse(forcing_c),
        "term_incompressibility_residual": _mse(incompressibility_residual),
        "term_vorticity_definition_residual": _mse(vorticity_definition_residual),
        "term_u_x": _mse(u_x_c),
        "term_v_y": _mse(v_y_c),
        "term_v_x": _mse(v_x_c),
        "term_u_y": _mse(u_y_c),
        "fd_dx": torch.as_tensor(sampler.du_phys, dtype=u.dtype, device=u.device),
        "fd_dy": torch.as_tensor(sampler.dv_phys, dtype=u.dtype, device=u.device),
    }


def pde_channel_order(dataset_type, num_channels=1):
    dataset_type = str(dataset_type).lower()
    if dataset_type == "ns" and int(num_channels) == 3:
        return ("u", "v", "omega")
    if dataset_type == "ocean" and int(num_channels) == 2:
        return ("u", "v")
    return None


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


def _sample_continuous_spatial_points(coord_uniques, sample_ratio, max_points, device):
    u_uni, v_uni, _ = coord_uniques
    total = int(torch.as_tensor(u_uni).numel()) * int(torch.as_tensor(v_uni).numel())
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
    return torch.stack([u_coords, v_coords], dim=-1)


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
    query_chunk_size = int(getattr(basis_function, "_pde_query_chunk_size", 0) or 0)
    q_total = int(sample_coords.shape[0])
    if q_total <= 0:
        channels = int(core_batch.shape[2]) if shared_basis_multichannel else 1
        return core_batch.new_empty(
            (int(core_batch.shape[0]), int(core_batch.shape[1]), 0, channels)
        )
    chunk_step = q_total if query_chunk_size <= 0 else min(query_chunk_size, q_total)
    outputs = []
    for start in range(0, q_total, chunk_step):
        end = min(start + chunk_step, q_total)
        coords_chunk = sample_coords[start:end]
        U = basis_function.mode_u(coords_chunk[:, 0])
        V = basis_function.mode_v(coords_chunk[:, 1])
        outputs.append(
            _sample_core_with_mode_basis_2d(
                core_batch, U, V, shared_basis_multichannel=shared_basis_multichannel
            )
        )
    return torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]


def _sample_core_with_mode_basis_3d(
    core_batch, factors, *, shared_basis_multichannel=False
):
    """Contract a per-time 3D Tucker core at arbitrary point factors.

    ``factors`` follows the model factor order (not necessarily physical
    ``u,v,w`` order; Ocean uses ``w,u,v``).
    """
    if len(factors) != 3:
        raise ValueError("3D basis query requires exactly three factor matrices.")
    a, b, c = factors
    if shared_basis_multichannel:
        if core_batch.dim() != 6:
            raise ValueError(
                f"Expected shared-basis 3D core (B,T,C,R1,R2,R3), got {tuple(core_batch.shape)}"
            )
        tmp = torch.einsum("btcijk,pi->btcpjk", core_batch, a)
        tmp = torch.einsum("btcpjk,pj->btcpk", tmp, b)
        out = torch.einsum("btcpk,pk->btcp", tmp, c)
        return out.permute(0, 1, 3, 2).contiguous()
    if core_batch.dim() != 5:
        raise ValueError(
            f"Expected scalar 3D core (B,T,R1,R2,R3), got {tuple(core_batch.shape)}"
        )
    tmp = torch.einsum("btijk,pi->btpjk", core_batch, a)
    tmp = torch.einsum("btpjk,pj->btpk", tmp, b)
    return torch.einsum("btpk,pk->btp", tmp, c).unsqueeze(-1)


def _sample_field_with_mode_basis_3d(
    basis_function,
    core_batch,
    sample_coords,
    *,
    spatial_axis_order=("u", "v", "w"),
    shared_basis_multichannel=False,
):
    """Query a per-time 3D basis/core at physical ``(u,v,w)`` points."""
    coords = torch.as_tensor(
        sample_coords, dtype=core_batch.dtype, device=core_batch.device
    )
    if coords.dim() != 2 or int(coords.shape[-1]) != 3:
        raise ValueError(
            f"3D basis query expects (P,3) physical coordinates, got {tuple(coords.shape)}"
        )
    if not hasattr(basis_function, "mode_w"):
        raise ValueError("3D basis query requires basis_function.mode_w.")
    order = tuple((str(axis).lower() for axis in spatial_axis_order))
    if set(order) != {"u", "v", "w"} or len(order) != 3:
        raise ValueError(f"Invalid spatial_axis_order={spatial_axis_order!r}")
    semantic = {"u": coords[:, 0], "v": coords[:, 1], "w": coords[:, 2]}
    modes = (basis_function.mode_u, basis_function.mode_v, basis_function.mode_w)
    chunk_size = int(getattr(basis_function, "_pde_query_chunk_size", 0) or 0)
    rank_product = int(core_batch.shape[-2]) * int(core_batch.shape[-1])
    channel_count = int(core_batch.shape[2]) if shared_basis_multichannel else 1
    batch_time = int(core_batch.shape[0]) * int(core_batch.shape[1])
    denom = max(1, batch_time * channel_count * rank_product)
    memory_safe_chunk = max(1, 32000000 // denom)
    if chunk_size <= 0:
        chunk_size = memory_safe_chunk
    else:
        chunk_size = min(chunk_size, memory_safe_chunk)
    if chunk_size <= 0 or int(coords.shape[0]) <= chunk_size:
        factors = tuple((mode(semantic[axis]) for mode, axis in zip(modes, order)))
        return _sample_core_with_mode_basis_3d(
            core_batch, factors, shared_basis_multichannel=shared_basis_multichannel
        )
    chunks = []
    for start in range(0, int(coords.shape[0]), chunk_size):
        end = min(start + chunk_size, int(coords.shape[0]))
        factors = tuple(
            (mode(semantic[axis][start:end]) for mode, axis in zip(modes, order))
        )
        chunks.append(
            _sample_core_with_mode_basis_3d(
                core_batch, factors, shared_basis_multichannel=shared_basis_multichannel
            )
        )
    return torch.cat(chunks, dim=2)


def build_dense_field_query_fn_3d(
    field,
    coord_u,
    coord_v,
    coord_w,
    *,
    spatial_axis_order=("u", "v", "w"),
    periodic_u=False,
    periodic_v=False,
    periodic_w=False,
):
    """Build a differentiable trilinear query function for channel-last 3D fields."""
    field_t = torch.as_tensor(field)
    squeeze_channel = False
    if field_t.dim() == 5:
        field_t = field_t.unsqueeze(-1)
        squeeze_channel = True
    if field_t.dim() != 6:
        raise ValueError(
            f"field must have shape (B,T,D,H,W) or (B,T,D,H,W,C), got {tuple(field_t.shape)}"
        )
    coords_by_axis = {
        "u": torch.as_tensor(
            coord_u, dtype=field_t.dtype, device=field_t.device
        ).reshape(-1),
        "v": torch.as_tensor(
            coord_v, dtype=field_t.dtype, device=field_t.device
        ).reshape(-1),
        "w": torch.as_tensor(
            coord_w, dtype=field_t.dtype, device=field_t.device
        ).reshape(-1),
    }
    order = tuple((str(x).lower() for x in spatial_axis_order))
    if len(order) != 3 or set(order) != {"u", "v", "w"}:
        raise ValueError(f"Invalid spatial_axis_order={spatial_axis_order!r}")
    coords = tuple((coords_by_axis[x] for x in order))
    shape = tuple((int(field_t.shape[2 + i]) for i in range(3)))
    if tuple((int(c.numel()) for c in coords)) != shape:
        raise ValueError(
            f"Dense field shape {shape} does not match coordinate lengths {tuple((int(c.numel()) for c in coords))}"
        )
    periodic = {"u": bool(periodic_u), "v": bool(periodic_v), "w": bool(periodic_w)}

    def _fractional(values, coord, axis):
        if int(coord.numel()) == 1:
            return torch.zeros_like(values)
        diffs = torch.diff(coord)
        positive = diffs[diffs > 0]
        step = (
            float(torch.median(positive).item())
            if positive.numel()
            else float((coord[-1] - coord[0]).item()) / max(int(coord.numel()) - 1, 1)
        )
        step = max(step, 1e-08)
        idx = (values - coord[0]) / step
        return (
            torch.remainder(idx, float(coord.numel()))
            if periodic[axis]
            else torch.clamp(idx, 0.0, float(coord.numel() - 1))
        )

    flat = field_t.reshape(
        int(field_t.shape[0]),
        int(field_t.shape[1]),
        int(torch.tensor(shape).prod()),
        int(field_t.shape[-1]),
    )

    def query_fn(query_coords):
        q = torch.as_tensor(query_coords, dtype=field_t.dtype, device=field_t.device)
        if q.dim() != 2 or int(q.shape[-1]) != 3:
            raise ValueError(
                f"query_coords must have shape (P,3), got {tuple(q.shape)}"
            )
        semantic = {"u": q[:, 0], "v": q[:, 1], "w": q[:, 2]}
        idx = [_fractional(semantic[a], coords[i], a) for i, a in enumerate(order)]
        base = [torch.floor(x).long() for x in idx]
        upper = [
            (
                torch.remainder(base[i] + 1, shape[i])
                if periodic[order[i]]
                else torch.clamp(base[i] + 1, max=shape[i] - 1)
            )
            for i in range(3)
        ]
        weights = [idx[i] - base[i].to(field_t.dtype) for i in range(3)]

        def gather(bits):
            inds = [upper[i] if bits[i] else base[i] for i in range(3)]
            linear = (
                inds[0] * shape[1] * shape[2] + inds[1] * shape[2] + inds[2]
            ).long()
            return flat.index_select(2, linear)

        out = None
        for mask in range(8):
            bits = tuple((mask >> i & 1 for i in range(3)))
            weight = torch.ones_like(weights[0])
            for i, bit in enumerate(bits):
                weight = weight * (weights[i] if bit else 1.0 - weights[i])
            term = gather(bits) * weight.reshape(1, 1, -1, 1)
            out = term if out is None else out + term
        return out[..., 0] if squeeze_channel else out

    return query_fn


class _FDQueryField3D:
    """Continuous 3D spatial query sampler with centered FD stencils."""

    def __init__(
        self,
        *,
        query_fn,
        sample_coords,
        coord_u,
        coord_v,
        coord_w,
        physical_length_u,
        physical_length_v,
        physical_length_w,
        periodic_u=False,
        periodic_v=False,
        periodic_w=False,
        fd_du=None,
        fd_dv=None,
        fd_dw=None,
        max_offset=1,
    ):
        coords = torch.as_tensor(sample_coords, dtype=torch.float32)
        if coords.dim() != 2 or int(coords.shape[-1]) != 3:
            raise ValueError(
                f"sample_coords must have shape (P,3), got {tuple(coords.shape)}"
            )
        self.query_fn = query_fn
        self.periodic = (bool(periodic_u), bool(periodic_v), bool(periodic_w))
        self.infos = []
        self.deltas_phys = []
        self.deltas_logical = []
        for coord, length, fd, periodic in zip(
            (coord_u, coord_v, coord_w),
            (physical_length_u, physical_length_v, physical_length_w),
            (fd_du, fd_dv, fd_dw),
            self.periodic,
        ):
            dphys, dlogical, info = _physical_to_logical_delta(
                fd, coord, length, periodic
            )
            self.deltas_phys.append(dphys)
            self.deltas_logical.append(dlogical)
            self.infos.append(info)
        self.base_coords = coords.to(dtype=torch.float32)
        for axis, (info, delta) in enumerate(zip(self.infos, self.deltas_logical)):
            if not self.periodic[axis]:
                self.base_coords[:, axis] = _FDQueryField2D._clamp_interior(
                    self.base_coords[:, axis], info, delta, max_offset
                )
        self.cache = {}

    def shifted_coords(self, *offsets):
        if len(offsets) != 3:
            raise ValueError("3D FD query requires three axis offsets.")
        out = self.base_coords.clone()
        for axis, offset in enumerate(offsets):
            if float(offset) == 0.0:
                continue
            values = out[:, axis] + float(offset) * self.deltas_logical[axis]
            if self.periodic[axis]:
                values = _periodic_shift_axis(
                    values, float(offset) * self.deltas_logical[axis], self.infos[axis]
                )
            else:
                values = torch.clamp(
                    values,
                    min=float(self.infos[axis]["coord_min"]),
                    max=float(self.infos[axis]["coord_max"]),
                )
            out[:, axis] = values
        return out

    def query(self, *offsets):
        key = tuple((float(x) for x in offsets))
        should_cache = not torch.is_grad_enabled()
        if should_cache and key in self.cache:
            return self.cache[key]
        out = self.query_fn(self.shifted_coords(*offsets))
        if out.dim() == 3:
            out = out.unsqueeze(-1)
        if out.dim() != 4:
            raise ValueError(
                f"3D query_fn must return (B,T,P) or (B,T,P,C), got {tuple(out.shape)}"
            )
        if should_cache:
            self.cache[key] = out
        return out


def _heat3d_nonuniform_time_diff_query(field, time_coords):
    t = torch.as_tensor(time_coords, dtype=field.dtype, device=field.device).reshape(-1)
    if int(field.shape[1]) != int(t.numel()) or int(t.numel()) < 3:
        raise ValueError(
            "Heat3D requires at least three frames and matching physical time coordinates."
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


def compute_heat3d_pde_loss_fd_query(
    *,
    query_fn,
    sample_coords,
    coord_u,
    coord_v,
    coord_w,
    time_coords_physical,
    Lx,
    Ly,
    Lz,
    alpha,
    fd_dx=None,
    fd_dy=None,
    fd_dz=None,
    fd_space_order=2,
    fd_time_order=2,
):
    """Heat equation residual from native continuous 3D queries."""
    sampler = _FDQueryField3D(
        query_fn=query_fn,
        sample_coords=sample_coords,
        coord_u=coord_u,
        coord_v=coord_v,
        coord_w=coord_w,
        physical_length_u=Lx,
        physical_length_v=Ly,
        physical_length_w=Lz,
        fd_du=fd_dx,
        fd_dv=fd_dy,
        fd_dw=fd_dz,
        max_offset=1,
    )
    dx, dy, dz = sampler.deltas_phys
    u = sampler.query(0.0, 0.0, 0.0)[..., 0]
    uxx = (
        sampler.query(+1.0, 0.0, 0.0)[..., 0]
        - 2.0 * u
        + sampler.query(-1.0, 0.0, 0.0)[..., 0]
    ) / dx**2
    uyy = (
        sampler.query(0.0, +1.0, 0.0)[..., 0]
        - 2.0 * u
        + sampler.query(0.0, -1.0, 0.0)[..., 0]
    ) / dy**2
    uzz = (
        sampler.query(0.0, 0.0, +1.0)[..., 0]
        - 2.0 * u
        + sampler.query(0.0, 0.0, -1.0)[..., 0]
    ) / dz**2
    lap = uxx + uyy + uzz
    u_t = _heat3d_nonuniform_time_diff_query(u, time_coords_physical)
    lap_c = lap[:, 1:-1]
    residual = u_t - float(alpha) * lap_c
    return {
        "pde": _mse(residual),
        "term_residual": _mse(residual),
        "term_u_t": _mse(u_t),
        "term_diffusion": _mse(float(alpha) * lap_c),
        "term_lap": _mse(lap_c),
        "fd_dx": torch.as_tensor(dx, dtype=u.dtype, device=u.device),
        "fd_dy": torch.as_tensor(dy, dtype=u.dtype, device=u.device),
        "fd_dz": torch.as_tensor(dz, dtype=u.dtype, device=u.device),
    }
