#include <metal_stdlib>
using namespace metal;

kernel void butterflies(device const float *input, device float *output,
                        constant uint &n, uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint width [[threads_per_threadgroup]]) {
    threadgroup float tile[4096];
    for (uint j = tid; j < n; j += width) tile[j] = input[row * n + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint gap = 1; gap < n; gap *= 2) {
        // Insert a zero bit at 'gap' in the pair number. Its partner differs
        // in that bit only, so no two threads overwrite the same butterfly.
        for (uint p = tid; p < n / 2; p += width) {
            uint left = 2 * (p & ~(gap - 1)) + (p & (gap - 1));
            float a = tile[left], b = tile[left + gap];
            tile[left] = a + b;
            tile[left + gap] = a - b;
        }
        // The next stage pairs different owners; everybody must finish first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = tid; j < n; j += width)
        output[row * n + j] = tile[j] * rsqrt(float(n));
}
