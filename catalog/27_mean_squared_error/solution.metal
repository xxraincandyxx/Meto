#include <metal_stdlib>
using namespace metal;

kernel void square_errors(device const float* predictions, device const float* targets,
                          device float* partial, constant uint& n,
                          uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint simds [[simdgroups_per_threadgroup]]) {
    // Spread each group's eight stripes across adjacent threads. Fuse
    // subtraction and squaring into the loads so only small partial sums
    // need a second GPU pass. Reading beyond the last stripe is guarded.
    float sum = 0;
    for (uint offset = group * 2048 + t; offset < min(n, (group + 1) * 2048); offset += 256) {
        float error = predictions[offset] - targets[offset];
        sum = fma(error, error, sum);
    }
    threadgroup float subtotal[64];
    float total = simd_sum(sum);
    if (!lane) subtotal[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float group_sum = 0;
        for (uint s = 0; s < simds; ++s) group_sum += subtotal[s];
        partial[group] = group_sum / float(n);
    }
}
