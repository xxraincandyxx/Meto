import torch.nn.functional as F


def solve(chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps,
          output, beta, B):
    # softplus(-margin) equals -log(sigmoid(margin)) but also handles a
    # strongly disfavored response without underflowing the sigmoid to zero.
    gap = (rejected_logps - chosen_logps).add_(chosen_ref_logps).sub_(rejected_ref_logps)
    output.copy_(F.softplus(gap.mul_(beta)).mean())
