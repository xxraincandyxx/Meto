import torch
import torch.nn.functional as F


def solve(Q, K, V, output, M, d, num_sinks, window_size):
    # All queries in a tile share the prefix and a nearby strip of K/V.
    # Exclude the gap before loading the keys, which bounds scratch memory
    # by tile * (sinks + window + tile), even for a very long sequence.
    device = Q.device
    for start in range(0, M, 128):
        stop = min(start + 128, M)
        first_local = max(num_sinks, start - window_size + 1)
        keys = torch.cat((torch.arange(min(num_sinks, stop), device=device),
                          torch.arange(min(first_local, stop), stop, device=device)))
        query = torch.arange(start, stop, device=device)[:, None]
        allowed = (keys <= query) & ((keys < num_sinks) | (keys > query - window_size))
        result = F.scaled_dot_product_attention(Q[None, None, start:stop],
                    K[keys][None, None], V[keys][None, None], attn_mask=allowed)
        output[start:stop].copy_(result[0, 0])
