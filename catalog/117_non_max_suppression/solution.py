import torch
from catalog.metal_shader import kernel


def solve(boxes, scores, keep, N, iou_threshold):
    # Stable sorting makes the original box index the tie breaker.
    order = scores.argsort(descending=True, stable=True)
    words = (N + 31) // 32
    masks = torch.empty((N, words), dtype=torch.int32, device=boxes.device)
    suppressed = torch.zeros(words, dtype=torch.int32, device=boxes.device)
    kernel(__file__, "overlaps")(boxes[order], masks, N, words, iou_threshold,
                                 threads=N * words)
    kernel(__file__, "select_boxes")(order.int(), masks, suppressed, keep, N, words,
                                      threads=256, group_size=256)
