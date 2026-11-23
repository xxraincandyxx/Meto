#include <metal_stdlib>
using namespace metal;

// 64 threads cooperate on a 32x32 output tile. A thread owns four rows and
// four neighboring columns. Its 16 accumulators survive across all inner tiles,
// so the output matrix is written just once.
kernel void multiply(device const float *a, device const float *b, device float *c,
                     constant uint &m, constant uint &k, constant uint &n,
                     uint2 group [[threadgroup_position_in_grid]],
                     uint2 local [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float left[32][16], right[16][32];
    uint row = group.y * 32, col = group.x * 32;
    float4 accum[4] = {};
    for (uint base = 0; base < k; base += 16) {
        // Read consecutive addresses together. Zero padding makes the same inner
        // loop valid for rectangular matrices and incomplete edge tiles.
        for (uint i = lane; i < 512; i += 64) {
            uint r = i / 16, t = i % 16;
            left[r][t] = row+r < m && base+t < k ? a[(row+r)*k+base+t] : 0;
            t = i / 32;
            uint j = i % 32;
            right[t][j] = base+t < k && col+j < n ? b[(base+t)*n+col+j] : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < 16; ++t) {
            float4 weights = *reinterpret_cast<threadgroup float4 *>(&right[t][local.x*4]);
            for (uint r = 0; r < 4; ++r)
                accum[r] = fma(float4(left[local.y*4+r][t]), weights, accum[r]);
        }
        // Everyone must finish reading these tiles before anyone replaces them.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 4; ++r) {
        uint y = row + local.y*4 + r, x = col + local.x*4;
        if (y < m) {
            for (uint j = 0; j < 4 && x+j < n; ++j) c[y*n+x+j] = accum[r][j];
        }
    }
}

#include <metal_simdgroup_matrix>

// On aligned shapes each SIMD group owns a 16x16 square. Metal distributes an
// 8x8 fragment across the lanes and executes a matrix multiply-accumulate.
// Four output fragments reuse the same two left and two right fragments.
// The host uses the guarded scalar-tile kernel above for ragged matrices.
kernel void multiplyFragments(device const float *a, device const float *b, device float *c,
                              constant uint &m, constant uint &k, constant uint &n,
                              uint2 group [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = group.y * 32 + (sg / 2) * 16;
    uint col = group.x * 32 + (sg % 2) * 16;
    simdgroup_float8x8 sums[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) sums[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0);
    for (uint inner = 0; inner < k; inner += 8) {
        simdgroup_float8x8 left[2], right[2];
        for (uint i = 0; i < 2; ++i) {
            simdgroup_load(left[i], a + (row+i*8)*k + inner, k);
            simdgroup_load(right[i], b + inner*n + col+i*8, n);
        }
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 2; ++j)
                simdgroup_multiply_accumulate(sums[i][j], left[i], right[j], sums[i][j]);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(sums[i][j], c + (row+i*8)*n + col+j*8, n);
}
