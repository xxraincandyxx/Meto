from catalog.metal_shader import kernel


def solve(positions, masses, forces, N):
    # A group owns one target particle. Its 32 lanes visit different sources
    # and reduce their force contributions, so even short rows use all lanes.
    kernel(__file__)(positions, masses, forces, N, threads=N * 32, group_size=32)
