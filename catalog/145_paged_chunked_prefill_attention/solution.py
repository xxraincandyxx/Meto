import torch
import torch.nn.functional as F


def solve(Q, K_cache, V_cache, block_table, context_lens, cu_seqlens_q, output,
          num_seqs, total_q, num_q_heads, num_kv_heads, head_dim, block_size, max_blocks_per_seq):
    # Only lengths guide host-side dispatch. K/V pages and their lookup table
    # stay on the GPU. Reassemble the live pages for one sequence at a time.
    boundaries = cu_seqlens_q.tolist()
    for seq, length in enumerate(context_lens.tolist()):
        start, stop = boundaries[seq:seq + 2]
        pages = block_table[seq, :(length + block_size - 1) // block_size].long()

        def lookup(cache):
            return cache[pages].flatten(0, 1)[:length].transpose(0, 1)[None]

        q = Q[start:stop].transpose(0, 1)[None]
        # The chunk queries are the *last* positions of the cached sequence.
        # Shift the causal diagonal right by the number of earlier cached tokens.
        qpos = torch.arange(stop - start, device=Q.device) + length - (stop - start)
        mask = qpos[:, None] >= torch.arange(length, device=Q.device)
        context = F.scaled_dot_product_attention(q, lookup(K_cache), lookup(V_cache),
                    attn_mask=mask, enable_gqa=num_q_heads != num_kv_heads)
        output[start:stop].copy_(context[0].transpose(0, 1))
