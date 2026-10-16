"""Shared Tucker decode utilities.

The project uses channel-last dense fields:
  - scalar 2D: (B,T,H,W,1)
  - shared-basis multichannel 2D: (B,T,H,W,C)
"""

import torch


def decode_from_basis_2d(U, V, core_batch):
    output = torch.einsum("mi, btij->btmj", U, core_batch)
    output = torch.einsum("nj, btmj->btmn", V, output)
    return output.unsqueeze(-1)


def decode_from_basis_2d_shared_channels(U, V, core_batch):
    output = torch.einsum("mi, btcij->btcmj", U, core_batch)
    output = torch.einsum("nj, btcmj->btcmn", V, output)
    return output.permute(0, 1, 3, 4, 2).contiguous()


def decode_from_basis_3d(U, V, W, core_batch):
    output = torch.einsum("di,btijk->btdjk", U, core_batch)
    output = torch.einsum("hj,btdjk->btdhk", V, output)
    output = torch.einsum("wk,btdhk->btdhw", W, output)
    return output.unsqueeze(-1)


def decode_from_basis_3d_shared_channels(U, V, W, core_batch):
    output = torch.einsum("di,btcijk->btcdjk", U, core_batch)
    output = torch.einsum("hj,btcdjk->btcdhk", V, output)
    output = torch.einsum("wk,btcdhk->btcdhw", W, output)
    return output.permute(0, 1, 3, 4, 5, 2).contiguous()


def decode_spatial_from_precomputed_basis(
    U,
    V,
    basis_extra,
    core_batch,
    shared_basis_multichannel=False,
    core_parameterization="per_time",
):
    if basis_extra is not None and isinstance(basis_extra, tuple):
        W = basis_extra[0]
        if shared_basis_multichannel:
            return decode_from_basis_3d_shared_channels(U, V, W, core_batch)
        return decode_from_basis_3d(U, V, W, core_batch)
    if shared_basis_multichannel:
        return decode_from_basis_2d_shared_channels(U, V, core_batch)
    return decode_from_basis_2d(U, V, core_batch)


def dense_decode_2d(model, core_batch, ind_input):
    model.mode = "training"
    U, V = model(input_ind_train=ind_input)
    return decode_from_basis_2d(U, V, core_batch)


def dense_decode_2d_shared_channels(model, core_batch, ind_input):
    model.mode = "training"
    U, V = model(input_ind_train=ind_input)
    return decode_from_basis_2d_shared_channels(U, V, core_batch)


def dense_decode_3d(model, core_batch, ind_input):
    model.mode = "training"
    U, V, W = model(input_ind_train=ind_input)
    return decode_from_basis_3d(U, V, W, core_batch)


def dense_decode_3d_shared_channels(model, core_batch, ind_input):
    model.mode = "training"
    U, V, W = model(input_ind_train=ind_input)
    return decode_from_basis_3d_shared_channels(U, V, W, core_batch)
