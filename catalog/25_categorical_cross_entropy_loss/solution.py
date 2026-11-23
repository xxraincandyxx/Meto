"""Reduce the negative log-probability of the labeled class."""

import torch


def solve(logits, true_labels, loss, N, C):
    # For each row, subtracting its maximum inside log-softmax keeps exp bounded.
    # The cross-entropy primitive selects the one true class per row and reduces
    # those losses. Labels stay on the GPU throughout; no scalar reads per example.
    loss.copy_(torch.nn.functional.cross_entropy(logits, true_labels.long(), reduction="mean"))
