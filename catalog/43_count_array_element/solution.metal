#include <metal_stdlib>
using namespace metal;

kernel void count_matches(device const int* values, device atomic_uint* answer,
                          constant uint& n, constant int& needle,
                          uint id [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    // Full-sized groups are dispatched so every SIMD lane reaches simd_sum.
    // Several coalesced stripes per group amortize the final global atomic.
    uint base = (id / 256) * 2048 + id % 256;
    uint local = 0;
    for (uint j = 0; j < 8; ++j) {
        uint i = base + j * 256;
        if (i < n) local += uint(values[i] == needle);
    }
    uint total = simd_sum(local);
    if (!lane && total) atomic_fetch_add_explicit(answer, total, memory_order_relaxed);
}
