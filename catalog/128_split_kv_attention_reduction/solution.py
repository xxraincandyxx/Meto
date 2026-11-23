import torch


def solve(partial_out, partial_lse, output, num_splits, num_heads, head_dim):
    # Each split already divided by its own softmax denominator. Combining
    # outputs therefore weights split s by exp(LSE_s) / sum_j exp(LSE_j).
    weights = partial_lse.softmax(dim=0).T.unsqueeze(1)
    by_head = partial_out.transpose(0, 1)
    output.copy_(torch.bmm(weights, by_head).squeeze(1))
