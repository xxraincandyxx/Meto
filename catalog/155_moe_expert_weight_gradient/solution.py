import torch


def solve(X, dY, group_offsets, dW, G, M_total, K, N):
    # Rows have already been bucketed by expert. Each weight gradient is the
    # product X_expert^T @ dY_expert, and views of the packed rows can be
    # consumed by Metal's GEMM directly. Read the short offset vector once.
    boundaries = group_offsets.tolist()
    for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            dW[expert].zero_()
        else:
            torch.mm(X[start:end].T, dY[start:end], out=dW[expert])
