import torch.nn.functional as F


def solve(Q, K, V, cu_seqlens, output, T, d, S):
    # Offsets are just S+1 integers. Read them together, so the host can launch
    # one correctly sized attention problem per sequence. Packing unrelated
    # sequences into one T-by-T score matrix would waste quadratic work.
    offsets = cu_seqlens.tolist()
    for start, end in zip(offsets, offsets[1:]):
        if start == end:
            continue
        q, k, v = (tensor[start:end][None, None] for tensor in (Q, K, V))
        # The causal flag supplies the triangular mask to the Metal backend.
        output[start:end].copy_(F.scaled_dot_product_attention(q, k, v, is_causal=True)[0, 0])
