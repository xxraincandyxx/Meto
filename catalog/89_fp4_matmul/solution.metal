#include <metal_stdlib>
using namespace metal;

constant float magnitude[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
inline float decode(uint nibble) { return (nibble & 8 ? -1 : 1) * magnitude[nibble & 7]; }

kernel void solve(
    device const uchar* packed,
    device const uchar* scales,
    device float2* out,
    uint i [[thread_position_in_grid]]) {
    // A byte holds two E2M1 values, high nibble first. Each block of 16
    // weights shares one E4M3 scale. Apply it while the pair is in registers.
    uint bits = packed[i], rawScale = scales[i / 8];
    uint exponent = (rawScale >> 3) & 15, mantissa = rawScale & 7;
    float scale = exponent ? ldexp(1.0f + float(mantissa) / 8, int(exponent) - 7)
                           : float(mantissa) / 512;
    if (rawScale & 128) scale = -scale;
    out[i] = scale * float2(decode(bits >> 4), decode(bits & 15));
}
