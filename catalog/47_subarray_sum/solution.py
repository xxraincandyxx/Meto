"""Sum an inclusive interval of a vector."""

import torch


def solve(input, output, N, S, E):
    # Slicing creates a view starting at S. E+1 is the exclusive end index,
    # so an interval containing just one element has length one. Keep all
    # accumulation in int32, the specified result type.
    output.copy_(torch.sum(input[S:E + 1], dtype=torch.int32))
