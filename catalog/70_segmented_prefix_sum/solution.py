import torch
from catalog.metal_shader import kernel


def scan(values, flags, result):
    count = values.numel()
    blocks = (count + 127) // 128
    totals = values.new_empty(blocks)
    resets = flags.new_empty(blocks)
    kernel(__file__, "local_scan")(values, flags, result, totals, resets, count, threads=blocks)
    if blocks > 1:
        # Compose block summaries recursively. A block containing a segment
        # head discards the preceding sum; a headless block adds its total.
        inclusive = torch.empty_like(totals)
        scan(totals, resets, inclusive)
        kernel(__file__, "carry")(result, flags, inclusive, count, threads=blocks)


def solve(values, flags, output, N):
    # Shifting first changes an inclusive segmented scan into an exclusive one.
    # A segment head contributes zero even when it occurs at a block boundary.
    shifted = torch.empty_like(values)
    shifted[0] = 0
    shifted[1:].copy_(values[:-1])
    shifted.masked_fill_(flags.bool(), 0)
    scan(shifted, flags, output)
