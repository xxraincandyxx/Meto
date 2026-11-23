#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a,
    device const float* x,
    device float* out,
    constant uint& length,
    uint i [[thread_position_in_grid]]) {
    // A sequence has a true time dependency. Keep its accumulator in one
    // register and place independent sequences on different grid lanes. This
    // trades time-axis parallelism for a single dispatch and O(1) scratch.
    uint start = i * length;
    float state = 0;
    for (uint t = 0; t < length; ++t) {
        state = fma(a[start + t], state, x[start + t]);
        out[start + t] = state;
    }
}
