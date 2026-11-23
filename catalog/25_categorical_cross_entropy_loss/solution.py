"""Reduce the negative log-probability of the labeled class."""

import torch

from catalog.metal_shader import kernel


def solve(logits, true_labels, loss, N, C):
    run = kernel(__file__, "row_loss")
    width = run.thread_execution_width
    rows = torch.empty(N, device=logits.device, dtype=logits.dtype)
    run(logits, true_labels, rows, C, threads=N * width, group_size=width)
    # Only one number per sample reaches the second reduction. The full
    # N-by-C matrix of log probabilities never touches device memory.
    torch.mean(rows, dim=0, out=loss.view(()))
