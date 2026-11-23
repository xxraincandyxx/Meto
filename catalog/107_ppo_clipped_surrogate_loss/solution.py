import torch


def solve(advantages, log_pi, log_pi_old, output, clip_eps, B, S):
    ratio = (log_pi - log_pi_old).exp_()
    # A positive advantage chooses the smaller ratio. A negative one flips
    # the order. Using the two one-sided bounds avoids forming two surrogates.
    bound = torch.where(advantages >= 0, ratio.clamp_max(1 + clip_eps),
                        ratio.clamp_min(1 - clip_eps))
    output.copy_((advantages * bound).mean().neg_())
