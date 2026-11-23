import torch


def solve(A, B, group_offsets, C, G, M_total, K, N):
    # Only G+1 integer boundaries are needed by the host for dispatch. Reading
    # them together incurs one synchronization. Each large product then runs as
    # a GPU GEMM on contiguous slices and writes directly to its assigned rows.
    offsets = group_offsets.tolist()
    for group, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
        if lo != hi:
            torch.mm(A[lo:hi], B[group], out=C[lo:hi])
