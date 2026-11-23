"""Sort unsigned words by eight stable four-bit radix passes."""

import torch

from catalog.metal_shader import kernel


def solve(input, output, N):
    groups = (N + 255) // 256
    counts = torch.empty(groups * 16, dtype=torch.int32, device=input.device)
    ends = torch.empty_like(counts)
    scratch = torch.empty_like(input)
    src = input
    # The least-significant digit is processed first. A stable scatter preserves
    # the order of lower digits as each more significant digit is added.
    for shift in range(0, 32, 4):
        dest = scratch if shift % 8 == 0 else output
        kernel(__file__, "digit_counts")(src, counts, N, shift, groups,
                                          threads=groups * 256, group_size=256)
        torch.cumsum(counts, dim=0, dtype=torch.int32, out=ends)
        kernel(__file__, "scatter_digit")(src, dest, ends, N, shift, groups,
                                           threads=groups * 256, group_size=256)
        src = dest
