import torch.nn.functional as F


def solve(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim):
    # Group consecutive query heads under their shared key/value head.
    # Folding the query-head axis into query positions reuses K and V without
    # expanding their buffers; each query row still normalizes independently.
    queries = Q.view(1, num_kv_heads, -1, head_dim)
    mix = F.scaled_dot_product_attention(queries, K[None], V[None])
    output.copy_(mix.reshape_as(output))
