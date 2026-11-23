#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const uchar* packed,
    device const half* scales,
    device float2* weights,
    constant uint& k,
    constant uint& groupSize,
    uint i [[thread_position_in_grid]]) {
    // The upper nibble encodes the even column. Unpack two weights together,
    // apply their scales while they are in registers, and write a contiguous
    // float2 for the GPU GEMM that follows.
    uint byteValue = packed[i];
    uint col = (2 * i) % k;
    uint group = (2 * i) / k * (k / groupSize) + col / groupSize;
    float2 values = float2(int(byteValue >> 4) - 8, int(byteValue & 15) - 8);
    weights[i] = values * float2(scales[group], scales[(2*i+1)/groupSize]);
}
