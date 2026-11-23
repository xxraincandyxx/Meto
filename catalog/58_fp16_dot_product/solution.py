import torch


def solve(A, B, result, N):
    # Widen the operands before multiplying. Half products can overflow
    # before positive and negative terms cancel in the final reduction.
    result.copy_(torch.dot(A.float(), B.float()))
