import torch


def solve(x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out,
          B, d_model, H_q, H_kv, D, S_max):
    packed = (x @ W_qkv).view(B, H_q + 2 * H_kv, D)
    # Rotate Q and K together in the packed projection. Pair the first and
    # second halves of each head, as required by this rotary embedding layout.
    left, right = packed[:, :H_q + H_kv].chunk(2, dim=-1)
    cosine, sine = cos_sin_cache[positions.long()][:, None].chunk(2, dim=-1)
    rotated = torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1)
    Q_out.copy_(rotated[:, :H_q])
    # Scatter writes one position per batch and head. All other cache entries
    # retain their previous values for the next decode step.
    slots = positions.long().view(B, 1, 1, 1).expand(B, H_kv, 1, D)
    K_cache.scatter_(2, slots, rotated[:, H_q:, None])
    V_cache.scatter_(2, slots, packed[:, H_q + H_kv:, None])
