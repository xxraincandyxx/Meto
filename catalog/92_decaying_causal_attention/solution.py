import torch


def solve(Q, K, V, output, seq_len, d_model, gamma):
    # Summarize earlier blocks with a decayed KᵀV matrix. Within a block,
    # explicitly form its small lower triangle. This avoids a global L-by-L
    # score matrix and combines all previous tokens with one GEMM per block.
    state = Q.new_zeros(d_model, d_model)
    scale = d_model ** -0.5
    for start in range(0, seq_len, 128):
        stop = min(start + 128, seq_len)
        length = stop - start
        pos = torch.arange(length, device=Q.device)
        distance = pos[:, None] - pos
        mask = torch.where(distance >= 0, gamma ** distance.clamp_min(0), 0.0)
        past = (Q[start:stop] @ state) * (gamma ** (pos + 1))[:, None]
        local = (Q[start:stop] @ K[start:stop].T) * mask
        output[start:stop].copy_((past + local @ V[start:stop]) * scale)
        decay = (gamma ** (length - 1 - pos))[:, None]
        state = torch.addmm(state, (K[start:stop] * decay).T, V[start:stop], beta=gamma ** length)
