"""Accumulate histograms locally before merging counts in device memory."""

from catalog.metal_shader import kernel


def solve(input, histogram, N, num_bins):
    histogram.zero_()
    groups = (N + 2047) // 2048
    kernel(__file__, "histogram_bins")(input, histogram, N, num_bins, threads=groups * 256, group_size=256)
