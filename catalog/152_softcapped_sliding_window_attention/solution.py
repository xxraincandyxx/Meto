import torch


def solve(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim,
          window_size, softcap):
    # Heads in the same group reuse the same K/V tensor. Tiling the query
    # dimension bounds the score matrix and skips keys outside all tile windows.
    group = num_q_heads // num_kv_heads
    q = Q.view(num_kv_heads, group, seq_len, head_dim)
    for begin in range(0, seq_len, 64):
        end = min(seq_len, begin + 64)
        first = max(0, begin - window_size + 1)
        k = K[:, None, first:end]
        scores = q[:, :, begin:end] @ k.transpose(-1, -2)
        scores.mul_(head_dim ** -0.5 / softcap).tanh_().mul_(softcap)
        queries = torch.arange(begin, end, device=Q.device)[:, None]
        keys = torch.arange(first, end, device=Q.device)
        mask = (keys > queries) | (keys <= queries - window_size)
        weights = scores.masked_fill_(mask, -torch.inf).softmax(-1)
        context = weights @ V[:, None, first:end]
        output[:, begin:end].copy_(context.reshape(num_q_heads, end - begin, head_dim))
