#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q,
    device const float* cosine,
    device const float* sine,
    device float* out,
    constant uint& width,
    uint i [[thread_position_in_grid]]) {
    // Pair the left half of each row with its right half. One lane loads
    // both partners once, then applies the two signed rotation formulas.
    uint halfWidth = width / 2;
    uint a = (i / halfWidth) * width + i % halfWidth;
    uint b = a + halfWidth;
    float x = q[a], y = q[b];
    out[a] = x * cosine[a] - y * sine[a];
    out[b] = y * cosine[b] + x * sine[b];
}
