import torch
import torch.nn.functional as F


def solve(Q, K_prefix, V_prefix, K_suffix, V_suffix, output, batch_size, num_heads, prefix_len, suffix_len, head_dim):
    # The suffix uses causal positions offset by the length of the shared
    # prefix. Build one mask for all heads and requests, then submit a
    # batched attention operation. Prefix expansion itself is a stride-zero view.
    keys = torch.cat((K_prefix.expand(batch_size, -1, -1, -1), K_suffix), dim=2)
    values = torch.cat((V_prefix.expand(batch_size, -1, -1, -1), V_suffix), dim=2)
    query_pos = torch.arange(suffix_len, device=Q.device) + prefix_len
    key_pos = torch.arange(prefix_len + suffix_len, device=Q.device)
    allowed = key_pos <= query_pos[:, None]
    output.copy_(F.scaled_dot_product_attention(Q, keys, values, attn_mask=allowed))
