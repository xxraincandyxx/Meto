"""Reduce squared prediction errors to a scalar mean."""

import torch


def solve(predictions, targets, mse, N):
    # Request a reduced loss directly. MPS handles the elementwise residual,
    # squaring and reduction, and the one-element destination receives the scalar.
    mse.copy_(torch.nn.functional.mse_loss(predictions, targets, reduction="mean"))
