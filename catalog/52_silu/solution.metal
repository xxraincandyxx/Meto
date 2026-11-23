#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x,
    device float* y,
    uint i [[thread_position_in_grid]]) {
    // One lane owns one scalar. Keeping the gate in a register avoids
    // writing a temporary sigmoid vector out to device memory.
    float value = x[i];
    y[i] = value / (1.0f + exp(-value));
}
