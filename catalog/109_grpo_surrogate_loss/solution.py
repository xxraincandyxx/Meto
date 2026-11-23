import torch
from catalog.metal_shader import kernel


def solve(rewards, log_pi, log_pi_old, log_ref, output, clip_eps, beta, B, G, S):
    # The reward normalization is tiny compared with the three token arrays.
    # Keep one advantage per response and let the kernel reuse it across S tokens.
    variance, center = torch.var_mean(rewards, dim=1, correction=0, keepdim=True)
    advantages = (rewards - center) / (variance.sqrt() + 1e-8)
    count = B * G * S
    groups = (count + 1023) // 1024
    sums = torch.empty((groups, 1), device=advantages.device)
    kernel(__file__, "grpo")(advantages, log_pi, log_pi_old, log_ref, sums, count,
                            clip_eps, beta, S, threads=256 * groups, group_size=256)
    torch.sum(sums, dim=0, out=output)
