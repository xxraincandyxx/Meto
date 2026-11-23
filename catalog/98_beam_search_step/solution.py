def solve(beam_scores, token_logprobs, new_beam_scores, parent_beam_indices, next_tokens, B, K, V):
    # Each beam can contribute at most K survivors to the global top K.
    # Prune locally before adding beam scores. The second top-k compares
    # K² candidates per batch instead of carrying K*V candidate sums.
    values, tokens = token_logprobs.topk(min(K, V), dim=-1)
    choices = values.size(-1)
    values.add_(beam_scores[:, :, None])
    best, indices = values.flatten(1).topk(K, dim=-1)
    new_beam_scores.copy_(best)
    parent_beam_indices.copy_(indices // choices)
    next_tokens.copy_(tokens.flatten(1).gather(1, indices))
