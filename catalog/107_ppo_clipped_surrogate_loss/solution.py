import torch
from catalog.metal_shader import kernel


def solve(advantages, log_pi, log_pi_old, output, clip_eps, B, S):
    # A positive advantage selects the smaller ratio and a negative one the larger.
    # Evaluate both the policy ratio and its clipping bound in GPU registers.
    count = B * S
    groups = (count + 1023) // 1024
    sums = torch.empty((groups, 1), device=advantages.device)
    kernel(__file__, "ppo")(advantages, log_pi, log_pi_old, sums, count,
                            clip_eps, threads=256 * groups, group_size=256)
    torch.sum(sums, dim=0, out=output)
