from catalog.metal_shader import kernel


def solve(positions, masses, forces, N):
    # Round the dispatch up to whole threadgroups. Inactive lanes in the last
    # group still take part in tile loads and both synchronization barriers.
    width = 128
    kernel(__file__)(positions, masses, forces, N, threads=((N + width - 1)//width)*width, group_size=width)
