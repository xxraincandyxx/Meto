#include <metal_stdlib>
using namespace metal;

kernel void row_loss(device const float* logits, device const int* labels, device float* losses,
                     constant uint& classes, uint id [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
    uint row = id / width;
    device const float* values = logits + row * classes;
    // The SIMD group shares one row. Subtract the same maximum in every
    // lane so even the largest exponential is only one.
    float local_max = -INFINITY;
    for (uint c = lane; c < classes; c += width) local_max = max(local_max, values[c]);
    float maximum = simd_max(local_max);
    float subtotal = 0;
    for (uint c = lane; c < classes; c += width) subtotal += exp(values[c] - maximum);
    float denominator = simd_sum(subtotal);
    // Subtract the target logit before adding the logarithm. This avoids
    // subtracting two almost-equal large numbers for an easy example.
    if (lane == 0) losses[row] = (maximum - values[labels[row]]) + log(denominator);
}
