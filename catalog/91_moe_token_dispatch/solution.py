import torch


def solve(x, expert_idx, dispatched_x, token_counts, T, D, E, capacity):
    # Stable sorting makes each expert's tokens contiguous while preserving
    # arrival order. Consecutive insertion offsets give all counts in one search.
    expert, permutation = torch.sort(expert_idx, stable=True)
    boundaries = torch.searchsorted(
        expert, torch.arange(E + 1, device=x.device, dtype=expert.dtype)
    )
    token_counts.copy_(boundaries.diff())
    rank = torch.arange(T, device=x.device) - boundaries[expert.long()]
    target = expert.long() * capacity + rank
    dispatched_x.view(-1, D).index_copy_(0, target, x.index_select(0, permutation))
