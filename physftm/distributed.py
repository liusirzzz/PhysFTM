"""Shared DDP (DistributedDataParallel) utilities.

Used by both train_spatial.py and the shared training utilities.
"""

import os
import torch
import torch.distributed as dist


def setup_distributed(default_device=None):
    """Initialise DDP if launched via torchrun, otherwise fall back to single-GPU.

    Args:
        default_device: torch.device to use in non-distributed mode.
            If *None*, defaults to ``cuda:0`` when CUDA is available, else ``cpu``.

    Returns:
        dict with keys: distributed, world_size, rank, local_rank, device.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        if default_device is None:
            default_device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        return {
            "distributed": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": default_device,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA in this script.")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return {
        "distributed": True,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": torch.device(f"cuda:{local_rank}"),
    }


def cleanup_distributed(ddp_state):
    """Tear down the process group if DDP is active."""
    if ddp_state["distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(ddp_state):
    """Return True on rank 0 (or in single-GPU mode)."""
    return ddp_state["rank"] == 0


def reduce_mean_scalar(value, ddp_state, device):
    """All-reduce a Python scalar across ranks and return the mean."""
    if not ddp_state["distributed"]:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= ddp_state["world_size"]
    return tensor.item()
