"""Mix value vectors for a separate query sequence in each attention head."""

import torch


def solve(Q, K, V, output, M, N, H, D):
    # The feature axis stays contiguous as we expose [1, heads, tokens, features].
    # The query and key sequence lengths can differ. The fused API handles
    # score scaling and numerically stable normalization over the N keys.
    def layout(x):
        return x.transpose(0, 1).unsqueeze(0)

    mixed = torch.nn.functional.scaled_dot_product_attention(layout(Q), layout(K), layout(V))
    output.copy_(mixed.squeeze(0).transpose(0, 1))
