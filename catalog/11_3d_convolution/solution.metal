#include <metal_stdlib>
using namespace metal;

kernel void filter_volume(device const float* volume, device const float* taps,
                          device float* result, constant uint& rows, constant uint& cols,
                          constant uint& kd, constant uint& kh, constant uint& kw,
                          uint3 dst [[thread_position_in_grid]]) {
    // X runs along the fastest-varying dimension. At each tap, neighboring
    // threads read neighboring voxels. Every launched position has a full window.
    float sum = 0.0f;
    for (uint dz = 0; dz < kd; ++dz)
        for (uint dy = 0; dy < kh; ++dy) {
            uint start = ((dst.z + dz) * rows + dst.y + dy) * cols + dst.x;
            uint tap = (dz * kh + dy) * kw;
            for (uint dx = 0; dx < kw; ++dx)
                sum = fma(volume[start + dx], taps[tap + dx], sum);
        }
    result[(dst.z * (rows - kh + 1) + dst.y) * (cols - kw + 1) + dst.x] = sum;
}
