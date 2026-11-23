import torch.nn.functional as F


def solve(Q, K, V, output, M, d):
    # Batch and head axes are singleton dimensions. The causal flag lets
    # Metal's attention backend exclude future keys without an explicit mask.
    values = F.scaled_dot_product_attention(
        Q[None, None], K[None, None], V[None, None], is_causal=True
    )
    output.copy_(values[0, 0])
