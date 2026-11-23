import torch
from catalog.metal_shader import kernel


def solve(logits, p, seed, sampled_token, vocab_size):
    # The stable sort preserves token order on ties. The exponential race can
    # use original probabilities because dividing by the same nucleus mass
    # leaves the winning token unchanged.
    probs, ids = torch.sort(logits.softmax(0), descending=True, stable=True)
    scores = torch.empty_like(probs)
    kernel(__file__)(probs, probs.cumsum(0), ids, p, seed, scores, threads=vocab_size)
    sampled_token.copy_(ids.gather(0, scores.argmax().view(1)))
