#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a,
    device const float* b,
    device float2* pairs,
    uint i [[thread_position_in_grid]]) {
    // Each lane owns an adjacent pair in the output, so two streams become
    // one packed stream in one dispatch. A float2 spans exactly two floats.
    pairs[i] = float2(a[i], b[i]);
}
