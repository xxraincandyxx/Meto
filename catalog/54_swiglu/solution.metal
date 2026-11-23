#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x,
    device float* y,
    constant uint& width,
    uint i [[thread_position_in_grid]]) {
    // The first half contains gates and the second contains values. Both
    // streams are contiguous across adjacent lanes; the gate stays in a register.
    float gate = x[i];
    y[i] = (gate / (1.0f + exp(-gate))) * x[i + width];
}
