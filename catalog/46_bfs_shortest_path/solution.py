"""Expand graph frontiers in parallel to find a shortest path on a grid."""

import torch

from catalog.metal_shader import kernel


def solve(grid, result, rows, cols, start_row, start_col, end_row, end_col):
    source = start_row * cols + start_col
    target = end_row * cols + end_col
    if rows * cols <= 4096:
        kernel(__file__, "local_bfs")(grid, result, rows, cols, source, target, threads=128, group_size=128)
        return
    device_frontiers(grid, result, rows, cols, source, target)


def device_frontiers(grid, result, rows, cols, source, target):
    if source == target:
        result.zero_()
        return
    distance = torch.full_like(grid, -1)
    distance[source] = 0
    frontier = torch.empty_like(grid)
    frontier[0] = source
    next_frontier = torch.empty_like(grid)
    tail = torch.empty_like(result)
    count, level = 1, 0
    while count:
        level += 1
        tail.zero_()
        kernel(__file__, "visit")(grid, frontier, next_frontier, distance, tail,
                                  rows, cols, level, threads=count)
        # The next launch needs only one host scalar. Node coordinates and
        # distances stay resident on Metal, and each cell enters a frontier once.
        count = tail.item()
        frontier, next_frontier = next_frontier, frontier
    result.copy_(distance[target:target + 1])
