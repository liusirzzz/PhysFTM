import numpy as np
import torch
from physftm.data.axes import ordered_spatial_values
from physftm.decoding.tucker import (
    decode_spatial_from_precomputed_basis,
    decode_from_basis_3d,
    decode_from_basis_3d_shared_channels,
)


def _decode_with_precomputed_basis(
    precomputed_basis,
    core_batch,
    shared_basis_multichannel=False,
    core_parameterization="per_time",
):
    if len(precomputed_basis) == 3 and core_batch.dim() in (5, 6):
        U, V, W = precomputed_basis
        return (
            decode_from_basis_3d_shared_channels(U, V, W, core_batch)
            if shared_basis_multichannel
            else decode_from_basis_3d(U, V, W, core_batch)
        )
    U, V = precomputed_basis[:2]
    return decode_spatial_from_precomputed_basis(
        U,
        V,
        None,
        core_batch,
        shared_basis_multichannel=shared_basis_multichannel,
        core_parameterization="per_time",
    )


def _query_coord_uniques_for_inference(
    ind_input, ind_uni, core_parameterization, device, spatial_axis_order=None
):
    u = torch.as_tensor(ind_uni[0], dtype=torch.float32, device=device).reshape(-1)
    v = torch.as_tensor(ind_uni[1], dtype=torch.float32, device=device).reshape(-1)
    if len(ind_uni) > 2 and int(np.asarray(ind_uni[2]).size) > 1:
        w = torch.as_tensor(ind_uni[2], dtype=torch.float32, device=device).reshape(-1)
        order = tuple(spatial_axis_order or ("u", "v", "w"))
        return ordered_spatial_values(order, u=u, v=v, w=w)
    return (u, v)


def _sampling_coord_uniques_for_inference(query_coord_uniques, core_parameterization):
    return query_coord_uniques[:2]


def _as_core_train_array(core_payload):
    core_np = np.asarray(
        core_payload.get("core_spatial", core_payload.get("core")), dtype=np.float32
    )
    if core_np.ndim not in (4, 5, 6):
        raise ValueError(f"Unsupported saved train core shape {core_np.shape}")
    return core_np


def _check_core_compatible(
    core_train_np, data, shared_basis_multichannel, core_parameterization="per_time"
):
    if int(core_train_np.shape[1]) != int(data.shape[1]):
        raise ValueError(
            f"Saved train core time length {core_train_np.shape[1]} != target data time length {data.shape[1]}"
        )
    if bool(shared_basis_multichannel):
        if core_train_np.ndim != (6 if data.dim() == 6 else 5):
            raise ValueError(
                f"Multichannel residual prior expects core shape (N,T,C,Ru,Rv), got {core_train_np.shape}"
            )
        if int(core_train_np.shape[2]) != int(data.shape[-1]):
            raise ValueError(
                f"Saved core channels {core_train_np.shape[2]} != data channels {data.shape[-1]}"
            )
    elif core_train_np.ndim != (5 if data.dim() == 6 else 4):
        raise ValueError(
            f"Scalar residual prior expects core shape (N,T,Ru,Rv), got {core_train_np.shape}"
        )
    return


def _mean_prior_core_batch(mean_core_np, batch_size, device):
    prior = torch.as_tensor(mean_core_np, dtype=torch.float32, device=device).unsqueeze(
        0
    )
    return prior.expand(int(batch_size), *prior.shape[1:]).contiguous()
