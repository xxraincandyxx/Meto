#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const packed_float3* pixels,
    device float* gray,
    uint i [[thread_position_in_grid]]) {
    // Image pixels have 12-byte spacing. packed_float3 honors that spacing;
    // ordinary float3 array elements would have a 16-byte stride.
    float3 color = float3(pixels[i]);
    gray[i] = dot(color, float3(0.299f, 0.587f, 0.114f));
}
