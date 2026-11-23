import torch
import torch.nn.functional as F


def solve(x, output, weights, cos, sin, seq_len):
    # The packed weights have output-major matrix layout. A single projection
    # produces all Q, K and V heads. Paired gates likewise share one GEMM.
    d, kv, wide = 512, 128, 1408
    norm1, qkv, proj, norm2, gate_up, down = weights.split(
        [d, (d + 2 * kv) * d, d * d, d, 2 * wide * d, d * wide]
    )
    normalized = F.rms_norm(x, (d,), norm1, 1e-5)
    projected = F.linear(normalized, qkv.view(d + 2 * kv, d))
    q, k, v = projected.split((d, kv, kv), dim=1)
    cosine, sine = cos[:, None], sin[:, None]

    def rotate(flat):
        first, second = flat.reshape(seq_len, -1, 64).chunk(2, dim=-1)
        rotated = torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)
        return rotated.transpose(0, 1)[None]

    query, key = rotate(q), rotate(k)
    value = v.reshape(seq_len, 2, 64).transpose(0, 1)[None]
    # Four query heads share each KV head. Broadcasting them through the
    # grouped attention primitive preserves causal sequence positions.
    attended = F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
    rows = attended[0].transpose(0, 1).reshape(seq_len, d)
    hidden = torch.addmm(x, rows, proj.view(d, d).T)
    normalized = F.rms_norm(hidden, (d,), norm2, 1e-5)
    gate, up = F.linear(normalized, gate_up.view(2 * wide, d)).chunk(2, dim=1)
    torch.addmm(hidden, F.silu(gate) * up, down.view(d, wide).T, out=output)
