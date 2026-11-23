"""Advance agents toward the average velocity of their local neighbors."""

from catalog.metal_shader import kernel


def solve(agents, agents_next, N):
    threads = ((N + 127) // 128) * 128
    kernel(__file__, "advance")(agents, agents_next, N, threads=threads, group_size=128)
