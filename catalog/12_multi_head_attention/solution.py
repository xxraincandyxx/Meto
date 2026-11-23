"""Evaluate all attention heads as a single batched attention operation."""

import torch


def solve(Q, K, V, output, N, d_model, h):
    # Split each token's feature vector into independent heads. Permuting the
    # strides exposes [batch, heads, sequence, head_width] to the MPS attention
    # dispatcher; it can select an implementation for the actual shape and device.
    def heads(x):
        return x.view(N, h, d_model // h).transpose(0, 1).unsqueeze(0)

    mixed = torch.nn.functional.scaled_dot_product_attention(heads(Q), heads(K), heads(V))
    output.view(N, h, -1).copy_(mixed.squeeze(0).transpose(0, 1))
