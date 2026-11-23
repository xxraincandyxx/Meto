"""Reduce squared prediction errors to a scalar mean."""

import torch

from catalog.metal_shader import kernel


def solve(predictions, targets, mse, N):
    groups = (N + 2047) // 2048
    partial = mse if groups == 1 else torch.empty(groups, device=mse.device, dtype=mse.dtype)
    kernel(__file__, "square_errors")(predictions, targets, partial, N, threads=groups * 256, group_size=256)
    if groups > 1:
        torch.sum(partial, dim=0, out=mse.view(()))
