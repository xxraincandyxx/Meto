import torch


def solve(A, B, C, BATCH, M, N, K):
    # bmm assigns independent matrix products to the batch dimension.
    # Widen before the dot products so copy_ rounds to half once per result.
    left = A.reshape(BATCH, M, K).float()
    right = B.reshape(BATCH, K, N).float()
    C.copy_(torch.bmm(left, right).reshape_as(C))
