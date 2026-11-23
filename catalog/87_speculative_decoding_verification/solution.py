import torch


def solve(draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V):
    # Gather the probability of each proposed token, then locate the earliest
    # failed test per sequence entirely on the GPU. T denotes full acceptance.
    proposal = draft_tokens.long().unsqueeze(-1)
    p = draft_probs.gather(-1, proposal).squeeze(-1)
    q = target_probs.gather(-1, proposal).squeeze(-1)
    step = torch.arange(T, device=p.device)
    failed = uniform_samples[:, :T] >= (q / p).clamp_max(1)
    stop = torch.where(failed, step, T).amin(1)
    output_tokens.zero_()
    output_tokens[:, :T].copy_(torch.where(step < stop[:, None], draft_tokens, 0))

    # Only the first rejected position needs a replacement distribution. If
    # every proposal passed, sample the bonus from the final target distribution.
    b = torch.arange(B, device=p.device)
    row = stop.clamp_max(T - 1)
    target = target_probs[b, row]
    residual = (target - draft_probs[b, row]).clamp_min_(0)
    mass = residual.sum(1, keepdim=True)
    adjusted = torch.where(mass > 0, residual / mass, 1 / V)
    cdf = torch.where((stop == T)[:, None], target, adjusted).cumsum(1)
    sample = torch.searchsorted(cdf, uniform_samples[:, -1:].contiguous()).clamp_max_(V - 1)
    output_tokens.scatter_(1, stop[:, None], sample.to(output_tokens.dtype))
