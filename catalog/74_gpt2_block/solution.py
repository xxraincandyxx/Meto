import torch
import torch.nn.functional as F


def solve(x, output, weights, seq_len):
    # Weights are packed back-to-back. split returns views and preserves the
    # input-by-output layout used by the four projections in this block.
    d, heads, wide = 768, 12, 3072
    n1, b1, qkv, qb, proj, pb, n2, b2, up, ub, down, db = weights.split(
        [d, d, d * 3 * d, 3 * d, d * d, d, d, d, d * wide, wide, wide * d, d]
    )
    z = F.layer_norm(x, (d,), n1, b1)
    components = torch.addmm(qb, z, qkv.view(d, 3 * d)).view(seq_len, 3, heads, d // heads)
    q, k, v = components.permute(1, 2, 0, 3).unbind(0)
    # This exercise specifies bidirectional attention. The sequence dimension
    # sits immediately before the feature dimension in the SDPA call.
    mix = F.scaled_dot_product_attention(q[None], k[None], v[None])[0]
    residual = x + torch.addmm(pb, mix.transpose(0, 1).reshape(seq_len, d), proj.view(d, d))
    normalized = F.layer_norm(residual, (d,), n2, b2)
    activated = F.gelu(torch.addmm(ub, normalized, up.view(d, wide)), approximate="tanh")
    torch.add(residual, torch.addmm(db, activated, down.view(wide, d)), out=output)
