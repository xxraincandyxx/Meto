"""Select the k largest values in descending order."""

import torch


def solve(input, output, N, k):
    # Ask MPS for a partial selection: only k results need to be ordered. Pass
    # the destination and an index scratch buffer to avoid another value copy.
    indices = torch.empty(k, dtype=torch.long, device=input.device)
    torch.topk(input, k, sorted=True, out=(output, indices))
