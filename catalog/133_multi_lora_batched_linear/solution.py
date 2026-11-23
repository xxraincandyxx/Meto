import torch


def solve(x, W, A, B, adapter_ids, y, T, K, N, R, L, scale):
    torch.mm(x, W, out=y)
    # Sorting by adapter lets all its tokens share one pair of low-rank GEMMs.
    # IDs of -1 sort before the first valid adapter and keep the base result.
    sorted_ids, order = torch.sort(adapter_ids)
    edges = torch.searchsorted(sorted_ids, torch.arange(L + 1, dtype=adapter_ids.dtype,
                                                       device=x.device)).tolist()
    for adapter, (start, end) in enumerate(zip(edges, edges[1:])):
        if end > start:
            tokens = order[start:end]
            delta = (x[tokens] @ A[adapter]) @ B[adapter]
            y.index_add_(0, tokens, delta, alpha=scale)
