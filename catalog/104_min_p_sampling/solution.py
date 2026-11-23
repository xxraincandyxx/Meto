import torch


def solve(logits, probs, min_p, B, V):
    # Dividing by the largest unnormalized probability is a subtraction in
    # log space. Shifting first also keeps the largest exponential exactly one.
    torch.sub(logits, logits.amax(1, keepdim=True), out=probs)
    probs.exp_()
    probs.masked_fill_(probs < min_p, 0)
    # Reuse the output allocation for all three stages. The maximum survives
    # even at min_p=1, so every row has a nonzero normalization constant.
    probs.div_(probs.sum(1, keepdim=True))
