#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* src,
    device float* dst,
    constant uint& height,
    constant uint& width,
    uint i [[thread_position_in_grid]]) {
    // Update a separate destination. Reading old values from the source
    // keeps one thread's update from becoming another thread's input.
    uint col = i % width, row = i / width;
    if (col == 0 || col + 1 == width || row == 0 || row + 1 == height) {
        dst[i] = src[i];
    } else {
        dst[i] = 0.25f * (src[i - width] + src[i + width] + src[i - 1] + src[i + 1]);
    }
}
