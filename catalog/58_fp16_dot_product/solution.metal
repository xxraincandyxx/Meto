#include <metal_stdlib>
using namespace metal;

kernel void solve(device const half* a, device const half* b, device float* partial,
                  constant uint& count, uint group [[threadgroup_position_in_grid]],
                  uint threadID [[thread_index_in_threadgroup]],
                  uint simdID [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0;
    for (uint stripe = 0; stripe < 8; ++stripe) {
        uint i = group * 2048 + stripe * 256 + threadID;
        // Neighboring lanes read neighboring halves. Convert in a register:
        // this saves two full float32 arrays and protects large products.
        if (i < count) sum = fma(float(a[i]), float(b[i]), sum);
    }
    float combined = simd_sum(sum);
    if (lane == 0) partial[group * 8 + simdID] = combined;
}
