#include <metal_stdlib>
using namespace metal;

kernel void activate(device const float *projected, device float *out,
                     constant uint &width, uint idx [[thread_position_in_grid]]) {
    uint gate = (idx / width) * (2 * width) + idx % width;
    float g = projected[gate];
    // The two reads cover contiguous runs for neighboring lanes. Writing
    // the product directly eliminates temporary sigmoid and SiLU buffers.
    out[idx] = (g / (1 + exp(-g))) * projected[gate + width];
}
