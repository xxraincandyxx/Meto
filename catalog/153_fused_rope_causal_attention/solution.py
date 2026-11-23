import torch
import torch.nn.functional as F


def solve(Q, K, V, cos, sin, output, num_q_heads, num_kv_heads, seq_len, head_dim):
    # Rotate each key once before broadcasting that key to its query-head
    # group. The lookup tables share one position axis across every head.
    def rotate(features):
        a, b = features.chunk(2, dim=-1)
        return torch.cat((torch.addcmul(a * cos, b, sin, value=-1),
                          torch.addcmul(b * cos, a, sin)), dim=-1)

    result = F.scaled_dot_product_attention(rotate(Q)[None], rotate(K)[None], V[None],
                 is_causal=True, enable_gqa=num_q_heads != num_kv_heads)
    output.copy_(result[0])
