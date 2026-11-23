import torch


def solve(logits, prompt_tokens, output_tokens, presence_penalty, frequency_penalty,
          repetition_penalty, output, B, V, P, G):
    # Accumulate generated counts first, since both additive penalties need
    # that count. Then extend it with the prompt to detect repeated tokens.
    counts = torch.zeros_like(logits)

    def accumulate(tokens):
        valid = tokens >= 0
        counts.scatter_add_(1, tokens.clamp_min(0).long(), valid.to(counts.dtype))

    accumulate(output_tokens)
    deduction = counts * frequency_penalty[:, None]
    deduction.add_((counts > 0) * presence_penalty[:, None])
    accumulate(prompt_tokens)
    penalty = repetition_penalty[:, None]
    adjusted = torch.where(logits > 0, logits / penalty, logits * penalty)
    output.copy_(torch.where(counts > 0, adjusted, logits) - deduction)
