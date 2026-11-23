import torch
import torch.nn.functional as F


def solve(Q, K, V, output, num_heads, seq_len, head_dim, block_size, num_selected):
    # A dot product is linear, so the mean token score ranks each block.
    # Reuse those individual dot products when normalizing attention later.
    scores = torch.bmm(K, Q.unsqueeze(2)).squeeze(2)
    pad = -seq_len % block_size
    by_block = F.pad(scores, (0, pad)).view(num_heads, -1, block_size)
    means = by_block.mean(2)
    if pad:
        means[:, -1].mul_(block_size / (block_size - pad))
    rank = means.argsort(dim=1, descending=True, stable=True)
    chosen = rank[:, :num_selected]
    slots = chosen[:, :, None] * block_size + torch.arange(block_size, device=K.device)
    slots = slots.flatten(1)
    safe = slots.clamp_max(seq_len - 1)
    selected = scores.gather(1, safe).mul_(head_dim ** -0.5)
    selected.masked_fill_(slots >= seq_len, -torch.inf)
    weights = selected.softmax(dim=1)
    values = V.gather(1, safe[:, :, None].expand(-1, -1, head_dim))
    output.copy_(torch.bmm(weights[:, None], values).squeeze(1))
