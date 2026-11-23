import torch
import torch.nn.functional as F


def solve(Q, K, V, output, M, N, d, alpha):
    # Softmax cancels any constant added to a whole row. The query-position
    # part of the bias vanishes, leaving just one N-element vector to broadcast.
    bias = torch.arange(N, device=Q.device, dtype=Q.dtype) * -alpha
    output.copy_(F.scaled_dot_product_attention(Q[None, None], K[None, None], V[None, None], attn_mask=bias.view(1, 1, 1, N))[0, 0])
