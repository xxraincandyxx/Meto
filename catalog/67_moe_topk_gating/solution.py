def solve(logits, topk_weights, topk_indices, M, E, k):
    # Select experts before exponentiating. Softmax only needs the k winning
    # logits per token, so its reduction and output both have width k.
    selected, ids = logits.topk(k, dim=1, sorted=True)
    topk_indices.copy_(ids)
    topk_weights.copy_(selected.softmax(1))
