import torch.nn.functional as F


def solve(x, residual, weight, out, N, C, eps):
    # Form the residual once. The RMS normalization primitive reduces each
    # contiguous row, applies its inverse RMS and the per-feature weight.
    out.copy_(F.rms_norm(x + residual, (C,), weight, eps))
