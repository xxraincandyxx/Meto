#include <metal_stdlib>
using namespace metal;

kernel void decode(device const uchar *raw, device const float *scales, device float *out,
                   uint i [[thread_position_in_grid]]) {
    uint byte = raw[i], power = (byte >> 3) & 15, fraction = byte & 7;
    // E4M3FN stores three fraction bits and an exponent biased by seven.
    // Subnormals omit the leading one and use the lowest normal exponent.
    float magnitude = power ? ldexp(1.0f + fraction * 0.125f, int(power) - 7)
                            : ldexp(float(fraction), -9);
    if (power == 15 && fraction == 7) magnitude = NAN;
    out[i] = ((byte & 128) ? -magnitude : magnitude) * scales[i / 128];
}
