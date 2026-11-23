#include <metal_stdlib>
using namespace metal;

kernel void reverse_pairs(device float* values, constant uint& n, uint i [[thread_position_in_grid]]) {
    // Only the first half launches. Each pair has exactly one writer and the
    // middle element of an odd-length array is already in the correct place.
    uint other = n - i - 1;
    float left = values[i];
    values[i] = values[other];
    values[other] = left;
}
