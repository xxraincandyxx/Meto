"""Scale and accumulate a half-precision GEMM with single-precision arithmetic."""

import torch


def solve(A, B, C, M, N, K, alpha, beta):
    # Keep the old C in the accumulation precision too. addmm gives the GPU a
    # matrix multiply with a scaled accumulation term, so matrix-sized Python
    # intermediates for alpha*(A@B) and beta*C can be avoided.
    accumulator = torch.addmm(C.float(), A.float(), B.float(), alpha=alpha, beta=beta)
    C.copy_(accumulator)  # copy_ performs the final float32 -> float16 rounding.
