"""Sort a float vector in ascending order on the Metal device."""

import torch


def solve(data, N):
    # MPS chooses its parallel ordering kernel by size and dtype. The sorted
    # values have their own storage, so copying them back satisfies the in-place
    # contract without readers racing with overwrites during the sort.
    ordered = torch.sort(data).values
    data.copy_(ordered)
