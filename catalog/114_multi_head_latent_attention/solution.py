import torch


def solve(q, kv_cache, W_UK, W_UV, output, num_heads, seq_len, kv_lora_rank,
          head_dim, rope_dim):
    # Move the key projection onto the (small) query side. Concatenating its
    # rotary coordinates builds one vector in the cache's native layout, so
    # a single GEMM scores the whole cache without expanding per-head keys.
    latent = torch.bmm(q[:, None, :head_dim], W_UK).squeeze(1)
    combined = torch.cat((latent, q[:, head_dim:]), dim=1)
    weights = (combined @ kv_cache.T).mul_(q.size(1) ** -0.5).softmax(1)
    compressed = weights @ kv_cache[:, :kv_lora_rank]
    output.copy_(torch.bmm(compressed[:, None], W_UV).squeeze(1))
