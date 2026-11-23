"""Multiply zero-point-centered int8 matrices and requantize the result."""

import torch


def solve(A, B, C, M, N, K, scale_A, scale_B, scale_C,
          zero_point_A, zero_point_B, zero_point_C):
    # Every int8 value can be represented exactly as a float. Centering before
    # matmul lets the optimized MPS GEMM perform the same float32 accumulation
    # as the problem specifies. Keep the specified rounding points on each side
    # of the scaling step: ties use round-to-nearest-even.
    left = A.view(M, K).float().sub_(zero_point_A)
    right = B.view(K, N).float().sub_(zero_point_B)
    acc = torch.mm(left, right).round_()
    acc.mul_(scale_A).mul_(scale_B).div_(scale_C).round_().add_(zero_point_C)
    C.view(M, N).copy_(acc.clamp_(-128, 127))
