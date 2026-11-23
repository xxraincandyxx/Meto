"""Estimate an integral by averaging the supplied function samples."""

import torch


def solve(y_samples, result, a, b, n_samples):
    # Each equally likely sample represents an interval of width (b-a)/n.
    # Reduce directly into the scalar destination, then apply interval width.
    torch.mean(y_samples, dim=0, out=result.view(()))
    result.mul_(b - a)
