import torch


def solve(A, B, C, M, N, K, nnz):
    # The wire format stores a dense array, including zeros. A contiguous GEMM
    # consumes it directly and reuses tiles of B across neighboring output rows.
    torch.mm(A.view(M, N), B.view(N, K), out=C.view(M, K))
