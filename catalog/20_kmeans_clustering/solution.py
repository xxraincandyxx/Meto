"""Fit planar k-means with parallel assignments and scatter reductions."""

import torch

from catalog.metal_shader import kernel


def solve(data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
          final_centroid_x, final_centroid_y, sample_size, k, max_iterations):
    final_centroid_x.copy_(initial_centroid_x)
    final_centroid_y.copy_(initial_centroid_y)
    ones = torch.ones_like(data_x)
    totals = torch.empty((3, k), device=data_x.device, dtype=data_x.dtype)
    for _ in range(max_iterations):
        kernel(__file__, "assign_points")(data_x, data_y, final_centroid_x, final_centroid_y,
                                          labels, k, threads=sample_size)
        totals.zero_()
        # A point contributes to one centroid. Scatter-add accumulates its count
        # and coordinates. Empty clusters keep their old location for the next pass.
        index = labels.long()
        for row, values in zip(totals, (data_x, data_y, ones)):
            row.index_add_(0, index, values)
        count = totals[2]
        occupied = count > 0
        final_centroid_x.copy_(torch.where(occupied, totals[0] / count, final_centroid_x))
        final_centroid_y.copy_(torch.where(occupied, totals[1] / count, final_centroid_y))
