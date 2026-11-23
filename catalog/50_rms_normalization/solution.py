"""Apply root-mean-square normalization with a scalar gain and bias."""

import torch

from catalog.metal_shader import kernel


def solve(input, gamma, beta, output, N, eps):
    # dot computes the sum of squares without an intermediate squared vector.
    # Leave its scalar on the GPU so the normalization can follow immediately.
    squared_norm = torch.dot(input, input)
    kernel(__file__, "normalize")(input, output, squared_norm, gamma, beta, eps, N, threads=N)
