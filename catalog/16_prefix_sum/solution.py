"""Find every inclusive prefix sum with a device scan."""

import torch


def solve(input, output, N):
    # Each result includes its own element. A parallel scan shares partial sums
    # within and across blocks; recomputing each prefix independently would read
    # the first element N times. Write the backend scan directly into the output.
    torch.cumsum(input, dim=0, out=output)
