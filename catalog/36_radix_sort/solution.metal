#include <metal_stdlib>
using namespace metal;

kernel void digit_counts(device const uint* keys, device int* counts, constant uint& n,
                         constant uint& shift, constant uint& groups, uint i [[thread_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint bins[16];
    if (t < 16) atomic_store_explicit(bins + t, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < n) atomic_fetch_add_explicit(bins + ((keys[i] >> shift) & 15), 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Digit-major layout makes all smaller digits, then earlier groups of the
    // same digit, precede this group in a single global prefix scan.
    if (t < 16) counts[t * groups + g] = atomic_load_explicit(bins + t, memory_order_relaxed);
}

kernel void scatter_digit(device const uint* keys, device uint* sorted, device const int* ends,
                          constant uint& n, constant uint& shift, constant uint& groups,
                          uint i [[thread_position_in_grid]], uint g [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                          uint simds [[simdgroups_per_threadgroup]]) {
    // Each SIMD group counts each digit and gives each lane its stable rank.
    // Subgroup counts then let lanes include matching keys in earlier subgroups.
    threadgroup uint totals[16][64];
    uint key = i < n ? keys[i] : 0;
    uint digit = i < n ? (key >> shift) & 15 : 16;
    uint rank = 0;
    for (uint d = 0; d < 16; ++d) {
        uint match = uint(digit == d);
        uint prefix = simd_prefix_exclusive_sum(match);
        uint total = simd_sum(match);
        if (match) rank = prefix;
        if (!lane) totals[d][simd] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= n) return;
    for (uint s = 0; s < simd; ++s) rank += totals[digit][s];
    uint block = digit * groups + g;
    uint start = block ? uint(ends[block - 1]) : 0;
    sorted[start + rank] = key;
}
