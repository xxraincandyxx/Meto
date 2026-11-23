import torch
from catalog.metal_shader import kernel


def solve(x, w_q, scales, y, M, N, K, group_size):
    # Decode once per weight and reuse the dense tile for every input row.
    # Float32 accumulation preserves the specified half output rounding.
    weights = torch.empty((N, K), device=x.device, dtype=torch.float32)
    kernel(__file__)(w_q, scales, weights, K, group_size, threads=w_q.numel())
    y.copy_(x.float() @ weights.T)
