import torch


def solve(rewards, log_pi, log_pi_old, log_ref, output, clip_eps, beta, B, G, S):
    # Each group shares its reward statistics across all its token positions.
    # Population variance gives zero for a group containing a single response.
    variance, center = torch.var_mean(rewards, dim=1, correction=0, keepdim=True)
    a = ((rewards - center) / (variance.sqrt() + 1e-8)).unsqueeze(-1)
    ratio = (log_pi - log_pi_old).exp_()
    clipped = torch.where(a >= 0, ratio.clamp_max(1 + clip_eps), ratio.clamp_min(1 - clip_eps))
    delta = log_ref - log_pi
    penalty = delta.expm1() - delta
    output.copy_((beta * penalty - a * clipped).mean())
