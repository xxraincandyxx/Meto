#include <metal_stdlib>
using namespace metal;

kernel void reduceSum(device const float *a, device float *partial, constant uint &n,
                      uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                      uint width [[threads_per_simdgroup]], uint groups [[simdgroups_per_threadgroup]]) {
    // First sum four stripes in registers. At each iteration neighboring threads
    // read neighboring elements, and one group consumes up to 1024 values.
    float local = 0;
    for (uint i = g*1024+t; i < min(n, (g+1)*1024); i += 256) local += a[i];
    // simd_sum communicates directly within the hardware execution group. Only
    // its leader publishes a subtotal, so the whole threadgroup needs one barrier.
    threadgroup float scratch[64];
    float subtotal = simd_sum(local);
    if (lane == 0) scratch[simd] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float combined = 0;
        for (uint j = lane; j < groups; j += width) combined += scratch[j];
        float total = simd_sum(combined);
        if (lane == 0) partial[g] = total;
    }
}
