#include <metal_stdlib>
using namespace metal;

kernel void local_scan(device const float* values, device const int* heads,
                       device float* result, device float* totals,
                       device int* reset, constant uint& count,
                       uint block [[thread_position_in_grid]]) {
    uint start = block * 128, end = min(start + 128, count);
    bool sawHead = false;
    float sum = 0, lost = 0;
    for (uint j = start; j < end; ++j) {
        if (heads[j]) { sawHead = true; sum = 0; lost = 0; }
        // Compensate for the low bits lost to rounding inside each block.
        float term = values[j] - lost;
        float updated = sum + term;
        lost = (updated - sum) - term;
        sum = updated;
        result[j] = sum;
    }
    totals[block] = sum;
    reset[block] = int(sawHead);
}

kernel void carry(device float* result, device const int* heads,
                  device const float* inclusive, constant uint& count,
                  uint block [[thread_position_in_grid]]) {
    if (block == 0) return;
    uint start = block * 128, end = min(start + 128, count);
    // After the first head, the local prefix already starts at zero.
    for (uint j = start; j < end && !heads[j]; ++j)
        result[j] += inclusive[block - 1];
}
