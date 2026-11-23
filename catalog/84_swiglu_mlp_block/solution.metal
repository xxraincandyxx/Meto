#include <metal_stdlib>
using namespace metal;

kernel void solve(device float* gate, device const float* up, uint i [[thread_position_in_grid]]) {
    float g = gate[i];
    gate[i] = g / (1.0f + exp(-g)) * up[i];
}
