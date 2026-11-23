#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device float* d,
    constant uint& n,
    constant uint& k,
    uint i [[thread_position_in_grid]]) {
    // For this pivot k, a cheaper i→j path may pass through k. The pivot's
    // own row and column stay unchanged (zero diagonal, no negative cycles).
    // Skipping their writes keeps the shared reads race-free within a pass.
    uint row = i / n, col = i % n;
    if (row != k && col != k)
        d[i] = min(d[i], d[row * n + k] + d[k * n + col]);
}
