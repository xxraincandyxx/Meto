import torch.nn.functional as F


def solve(Q, K, V, lambdas, output, seq_len, num_heads, head_dim, lambda_init):
    q1, q2 = Q.transpose(0, 1)[None].chunk(2, -1)
    k1, k2 = K.transpose(0, 1)[None].chunk(2, -1)
    values = V.transpose(0, 1)[None]
    # Linearity in V allows us to subtract the two attention outputs.
    # Each call normalizes its own scores and can use native causal attention.
    first = F.scaled_dot_product_attention(q1, k1, values, is_causal=True)
    second = F.scaled_dot_product_attention(q2, k2, values, is_causal=True)
    mixed = first - second * lambdas[None, :, None, None]
    rms = (mixed.square().mean(dim=-1, keepdim=True) + 1e-5).rsqrt_()
    output.copy_((mixed * rms * (1 - lambda_init))[0].transpose(0, 1))
