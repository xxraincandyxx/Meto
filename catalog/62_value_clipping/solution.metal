#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x,
    device float* y,
    constant float& low,
    constant float& high,
    uint i [[thread_position_in_grid]]) {
    // Neighboring lanes visit neighboring floats. clamp maps the two
    // boundaries onto min/max instructions in the same memory pass.
    y[i] = clamp(x[i], low, high);
}
