def solve(src, index, output, N, D, M):
    # Different source rows can select the same destination. index_add_ is
    # a collision-safe accumulation across rows; all D features of a row
    # reuse its index. Unreferenced rows must still be reset to zero.
    output.zero_()
    output.index_add_(0, index, src)
