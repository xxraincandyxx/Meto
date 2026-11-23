import torch
import torch.nn.functional as F


def solve(x, topk_ids, topk_weights, w_gate, w_up, w_down, output, T, H, I, E, K):
    # Group selected routes by expert. An expert's weights are read once by a
    # matrix multiply on its incoming tokens. Repeated routes are added back
    # to their source token, including multiple choices of the same expert.
    sorted_ids, permutation = torch.sort(topk_ids.flatten())
    boundaries = torch.searchsorted(sorted_ids, torch.arange(E + 1, device=x.device,
                                                            dtype=sorted_ids.dtype)).tolist()
    weights = topk_weights.flatten()
    output.zero_()
    for expert, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            continue
        routes = permutation[start:end]
        tokens = routes // K
        chosen = x[tokens]
        hidden = F.silu(chosen @ w_gate[expert]) * (chosen @ w_up[expert])
        response = (hidden @ w_down[expert]) * weights[routes, None]
        output.index_add_(0, tokens, response)
