import torch
import torch.nn.functional as F


def solve(x, gamma, beta, W, bias, y, M, K, N, eps):
    # Layer normalization shares row statistics across all features. addmm
    # then applies the projection with a broadcast bias in its GEMM epilogue.
    normalized = F.layer_norm(x, (K,), gamma, beta, eps)
    projected = torch.addmm(bias, normalized, W)
    y.copy_(F.gelu(projected, approximate="none"))
