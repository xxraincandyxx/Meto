import torch
from catalog.metal_shader import kernel


def solve(x_q, x_scales, w_q, w_scales, alpha, y, M, N, K):
    # Decode both packed matrices in parallel, then reuse each decoded weight
    # across the GEMM. Apply the global scale before rounding once to fp16.
    left = torch.empty((M, K), device=y.device, dtype=torch.float32)
    right = left.new_empty(N, K)
    decode = kernel(__file__)
    decode(x_q, x_scales, left, threads=x_q.numel())
    decode(w_q, w_scales, right, threads=w_q.numel())
    y.copy_((left @ right.T).mul_(alpha))
