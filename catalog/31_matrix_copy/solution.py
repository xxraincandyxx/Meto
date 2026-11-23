"""Copy contiguous matrix storage on device."""

import torch


def solve(A, B, N):
    # A copy has no arithmetic to hide memory latency. Hand the entire region to
    # the backend's bulk copy path so it can use a device transfer operation.
    B.copy_(A)
