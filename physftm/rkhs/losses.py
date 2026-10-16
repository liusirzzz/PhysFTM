"""Reconstruction losses and sparse observation helpers for RKHS FTM."""

from __future__ import annotations
import torch

Tensor = torch.Tensor


def build_sparse_observation_index(mask: Tensor):
    indices = torch.where(mask > 0)
    return tuple((index.long() for index in indices))
