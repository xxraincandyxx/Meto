import torch
from catalog.metal_shader import kernel


def solve(chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output, beta, B):
    # Each preference pair contributes one loss. Four contiguous streams feed
    # the fused margin, softplus and sum, leaving only partial sums in memory.
    count = B
    groups = (count + 1023) // 1024
    sums = torch.empty((groups, 1), device=chosen_logps.device)
    kernel(__file__, "dpo")(chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, sums, count,
                            beta, threads=256 * groups, group_size=256)
    torch.sum(sums, dim=0, out=output)
