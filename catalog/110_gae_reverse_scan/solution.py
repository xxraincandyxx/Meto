import torch
from catalog.metal_shader import kernel


def solve(rewards, values, advantages, gamma, lam, B, S):
    # Process independent 64-step chunks first. This exposes work even when the
    # full series is long and the batch contains only a few trajectories.
    chunks = (S + 63) // 64
    totals = torch.empty((B, chunks), device=values.device)
    decay = gamma * lam
    kernel(__file__, "local_scan")(rewards, values, advantages, totals, S, chunks,
                                   gamma, decay, threads=B * chunks)
    if chunks > 1:
        kernel(__file__, "carry_scan")(totals, chunks, decay ** 64, threads=B)
        kernel(__file__, "apply_carry")(advantages, totals, S, chunks, decay, threads=B * S)
