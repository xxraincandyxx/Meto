import torch.nn.functional as F


def solve(Q, K, V, output, T, S, H, D):
    # The input interleaves heads at each time step. Reorder dimensions with
    # views, allowing the native scaled-dot-product backend to batch all heads.
    # Every query may see the full cache, including other query positions.
    q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (Q, K, V))
    context = F.scaled_dot_product_attention(q, k, v)
    output.copy_(context[0].transpose(0, 1))
