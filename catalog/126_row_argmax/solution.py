import torch


def solve(logits, output, B, V):
    # The row reduction carries the column index of the winning value.
    # argmax returns the first column on ties, so no second pass over an
    # equality mask or B-by-V array of column numbers is needed.
    output.copy_(torch.argmax(logits, dim=1))
