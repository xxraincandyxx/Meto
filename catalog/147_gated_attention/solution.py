import torch.nn.functional as F


def solve(Q, K, V, X, W_g, output, M, d_model, h):
    # Splitting the hidden dimension into heads is a view. A single batched
    # attention invocation processes all heads and enforces causality.
    q, k, v = (t.view(M, h, -1).transpose(0, 1)[None] for t in (Q, K, V))
    context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    gate = (X @ W_g).sigmoid_()
    output.copy_(context[0].transpose(0, 1).reshape_as(X) * gate)
