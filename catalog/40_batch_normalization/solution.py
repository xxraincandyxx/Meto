"""Normalize each feature over samples, then apply channel scale and bias."""

import torch


def solve(input, gamma, beta, output, N, C, eps):
    # A joint reduction obtains mean and population variance over the batch.
    # Once per channel, derive an affine slope. Keep the final computation
    # centered before scaling to avoid cancellation when the mean is large.
    variance, mean = torch.var_mean(input.view(N, C), dim=0, correction=0)
    slope = gamma * torch.rsqrt(variance + eps)
    output.copy_(torch.addcmul(beta, input - mean, slope))
