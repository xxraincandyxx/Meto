#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x,
    device const float* weight,
    device const float* bias,
    device float* out,
    constant uint& length,
    constant uint& channels,
    constant uint& taps,
    uint i [[thread_position_in_grid]]) {
    // Channels are independent. Adjacent lanes read adjacent channels at
    // the same time step, so the native B,L,D layout can stay untouched.
    // Missing history at the left boundary is implicitly zero.
    uint channel = i % channels, t = (i / channels) % length;
    float value = bias[channel];
    for (uint lag = 0; lag < min(t + 1, taps); ++lag)
        value = fma(x[i - lag * channels], weight[channel * taps + lag], value);
    out[i] = value;
}
