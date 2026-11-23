#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x,
    device const int* rank,
    device float* out,
    uint i [[thread_position_in_grid]]) {
    // A prefix count gives every survivor an increasing, unique destination
    // without locks. Slots beyond the last surviving value remain untouched.
    if (x[i] > 0) out[rank[i] - 1] = x[i];
}
