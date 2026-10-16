"""Evaluation helpers for RKHS FTM training."""

import torch
import torch.distributed as dist
from physftm.artifacts.loading import inverse_normalize_tensor


def evaluate(model, loader, ind_input, ddp_state, device, norm_stats):
    model.eval()
    rmse_sum = torch.tensor(0.0, device=device)
    mae_sum = torch.tensor(0.0, device=device)
    rel_l2_num_sum = torch.tensor(0.0, device=device)
    rel_l2_den_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for data, _mask, batch_ind in loader:
            data = data.to(device, non_blocking=True)
            batch_ind = batch_ind.to(device, non_blocking=True)
            output = (
                model.module.dense_decode(batch_ind, ind_input)
                if ddp_state["distributed"]
                else model.dense_decode(batch_ind, ind_input)
            )
            output_phys = inverse_normalize_tensor(output, norm_stats)
            data_phys = inverse_normalize_tensor(data, norm_stats)
            diff = output_phys - data_phys
            rmse_sum += torch.sqrt(torch.mean(diff**2))
            mae_sum += torch.mean(torch.abs(diff))
            rel_l2_num_sum += torch.sum(diff**2)
            rel_l2_den_sum += torch.sum(data_phys**2)
            count += 1
    if ddp_state["distributed"]:
        dist.all_reduce(rmse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mae_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(rel_l2_num_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(rel_l2_den_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    rel_l2 = torch.sqrt(rel_l2_num_sum / rel_l2_den_sum.clamp_min(1e-12))
    return ((rmse_sum / count).item(), (mae_sum / count).item(), rel_l2.item())
