import torch
import torch.nn.functional as F
from catalog.metal_shader import kernel


def solve(Q, K_cache, V_cache, K_tree, V_tree, parents, output, P, T, H, D):
    allowed = torch.ones((T, P + T), dtype=torch.bool, device=Q.device)
    kernel(__file__, "lineage")(parents, allowed, P, T, threads=T)
    # Keys in the verified prefix are visible to all branches. Only the
    # speculative suffix needs an explicit ancestry check.
    k = torch.cat((K_cache, K_tree), dim=0).transpose(0, 1)[None]
    v = torch.cat((V_cache, V_tree), dim=0).transpose(0, 1)[None]
    result = F.scaled_dot_product_attention(Q.transpose(0, 1)[None], k, v, attn_mask=allowed)
    output.copy_(result[0].transpose(0, 1))
