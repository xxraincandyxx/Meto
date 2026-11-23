import torch.nn.functional as F


def solve(X, Y, N):
    # Exact GELU uses the Gaussian cumulative distribution. The native Metal
    # activation evaluates it pointwise for the entire contiguous allocation.
    Y.copy_(F.gelu(X, approximate="none"))
