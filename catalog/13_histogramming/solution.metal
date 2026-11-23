#include <metal_stdlib>
using namespace metal;

kernel void histogram_bins(device const int* values, device atomic_uint* out,
                           constant uint& n, constant uint& bins,
                           uint tid [[thread_index_in_threadgroup]],
                           uint block [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint local[1024];  // Upper bound in the problem.
    for (uint b = tid; b < bins; b += 256) atomic_store_explicit(local + b, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // One group consumes 2048 samples in coalesced stripes. Contended merges
    // into the global histogram happen once per occupied bin per group.
    for (uint i = block * 2048 + tid; i < min(n, (block + 1) * 2048); i += 256) {
        uint b = uint(values[i]);
        if (b < bins) atomic_fetch_add_explicit(local + b, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint b = tid; b < bins; b += 256) {
        uint count = atomic_load_explicit(local + b, memory_order_relaxed);
        if (count) atomic_fetch_add_explicit(out + b, count, memory_order_relaxed);
    }
}
