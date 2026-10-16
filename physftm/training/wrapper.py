"""RKHS FTM wrapper around basis modules and the per-sample core table."""

import torch
from torch import nn
from physftm.decoding.tucker import (
    dense_decode_2d,
    dense_decode_2d_shared_channels,
    dense_decode_3d,
    dense_decode_3d_shared_channels,
)


class RKHSFTMWrapper(nn.Module):

    def __init__(
        self,
        basis_function,
        tucker_core,
        shared_basis_multichannel=False,
        core_table_on_cpu=False,
        core_parameterization="per_time",
    ):
        super().__init__()
        self.basis_function = basis_function
        self.core_table_on_cpu = bool(core_table_on_cpu)
        self.core_parameterization = str("per_time").lower()
        if self.core_table_on_cpu:
            self.tucker_core = tucker_core.detach().cpu().contiguous()
        else:
            self.tucker_core = nn.Parameter(tucker_core)
        self.shared_basis_multichannel = bool(shared_basis_multichannel)

    def compute_device(self):
        return next(self.basis_function.parameters()).device

    def get_core_batch(self, batch_ind, *, detach=False, device=None):
        device = self.compute_device() if device is None else torch.device(device)
        if self.core_table_on_cpu:
            idx = batch_ind.detach().cpu().long()
            core_batch = self.tucker_core.index_select(0, idx).to(
                device=device, non_blocking=True
            )
        else:
            core_batch = self.tucker_core[batch_ind.to(device=self.tucker_core.device)]
        return core_batch.detach() if detach else core_batch

    def put_core_batch(self, batch_ind, core_batch):
        if self.core_table_on_cpu:
            idx = batch_ind.detach().cpu().long()
            self.tucker_core.index_copy_(0, idx, core_batch.detach().cpu())
        else:
            self.tucker_core.data[batch_ind.to(device=self.tucker_core.device)] = (
                core_batch.detach()
            )

    def dense_decode(self, batch_ind, ind_input):
        core_batch = self.get_core_batch(batch_ind)
        if getattr(self.basis_function, "spatial_dim", 2) == 3:
            if self.shared_basis_multichannel:
                return dense_decode_3d_shared_channels(
                    self.basis_function, core_batch, ind_input
                )
            return dense_decode_3d(self.basis_function, core_batch, ind_input)
        if self.shared_basis_multichannel:
            return dense_decode_2d_shared_channels(
                self.basis_function, core_batch, ind_input
            )
        return dense_decode_2d(self.basis_function, core_batch, ind_input)


def reconstruction_loss_from_output(
    output, data_batch, mask_batch=None, reduction="rmse"
):
    diff = output - data_batch
    if mask_batch is not None:
        if mask_batch.shape != diff.shape:
            if (
                mask_batch.dim() == diff.dim()
                and int(mask_batch.shape[-1]) == 1
                and (int(diff.shape[-1]) > 1)
                and (tuple(mask_batch.shape[:-1]) == tuple(diff.shape[:-1]))
            ):
                mask_batch = mask_batch.expand(
                    *mask_batch.shape[:-1], int(diff.shape[-1])
                )
            else:
                raise ValueError(
                    f"mask_batch shape {tuple(mask_batch.shape)} is incompatible with diff shape {tuple(diff.shape)}"
                )
        diff = diff[mask_batch > 0]
    if reduction == "rmse":
        return torch.sqrt(torch.mean(diff**2))
    if reduction == "mse":
        return torch.mean(diff**2)
    if reduction == "mae":
        return torch.mean(torch.abs(diff))
    raise ValueError(f"Unknown reduction: {reduction}")


def broadcast_mask_channels(mask, ref):
    if mask is None:
        return None
    if mask.shape == ref.shape:
        return mask
    if (
        mask.dim() == ref.dim()
        and int(mask.shape[-1]) == 1
        and (int(ref.shape[-1]) > 1)
        and (tuple(mask.shape[:-1]) == tuple(ref.shape[:-1]))
    ):
        return mask.expand(*mask.shape[:-1], int(ref.shape[-1]))
    raise ValueError(
        f"mask shape {tuple(mask.shape)} is incompatible with ref shape {tuple(ref.shape)}"
    )
