import torch
import torch.nn.functional as F


def solve(Q, K, V, output, M, d, window_size):
    # Each block of query rows only needs the keys within a window of that
    # block. A short rectangular mask clips the two triangular ends. Peak
    # score storage depends on block and window width, independently of M².
    positions = torch.arange(M, device=Q.device)
    for start in range(0, M, 128):
        end = min(start + 128, M)
        left, right = max(0, start - window_size), min(M, end + window_size)
        allowed = (positions[start:end, None] - positions[None, left:right]).abs() <= window_size
        output[start:end].copy_(F.scaled_dot_product_attention(
            Q[None, None, start:end], K[None, None, left:right], V[None, None, left:right],
            attn_mask=allowed
        )[0, 0])
