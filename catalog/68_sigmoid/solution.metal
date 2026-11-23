#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x,
    device float* y,
    uint i [[thread_position_in_grid]]) {
    // exp(-abs(x)) lies in [0, 1], even for very large negative logits.
    // The two branches share an exponential and a denominator.
    float value = x[i];
    float e = exp(-abs(value));
    y[i] = (value < 0 ? e : 1.0f) / (1.0f + e);
}
