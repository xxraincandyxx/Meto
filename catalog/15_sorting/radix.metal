#include <metal_stdlib>
using namespace metal;

kernel void encode(device const float* x, device uint* keys, uint i [[thread_position_in_grid]]) {
    // For positives, flipping the sign bit moves them above all negatives.
    // Negative IEEE magnitudes increase in the opposite order: invert every
    // bit to make increasing unsigned keys correspond to increasing values.
    uint bits = as_type<uint>(x[i]);
    bool nan_value = (bits & 0x7fffffffu) > 0x7f800000u;
    keys[i] = nan_value ? 0xffffffffu : bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
}

kernel void decode(device uint* values, uint i [[thread_position_in_grid]]) {
    uint key = values[i];
    values[i] = key ^ ((key >> 31) ? 0x80000000u : 0xffffffffu);
}
