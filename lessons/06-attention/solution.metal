#include <metal_stdlib>
using namespace metal;

// One hardware SIMD group streams a query through the keys. Lanes cooperate on
// each dot product and retain separate output features in registers. The score
// matrix would consume M*N floats; this kernel writes only the final M*d values.
kernel void attend(device const float *q, device const float *k, device const float *v,
                   device float *out, constant uint &n, constant uint &d,
                   uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                   uint width [[threads_per_simdgroup]]) {
    // d <= 128 for this lesson. Eight strips cover SIMD widths of 16 and above.
    float query[8], numerator[8] = {};
    for (uint f = lane; f < d; f += width) query[f/width] = q[row*d+f];
    float high = -INFINITY, denominator = 0, scale = rsqrt(float(d));
    for (uint j = 0; j < n; ++j) {
        float dot = 0;
        for (uint f = lane; f < d; f += width) dot = fma(query[f/width], k[j*d+f], dot);
        float score = simd_sum(dot) * scale;
        // A larger score changes the exponential scale. Rescale all previous
        // contributions before inserting this key; no second softmax pass is needed.
        float next = max(high, score);
        float oldWeight = exp(high-next), weight = exp(score-next);
        denominator = fma(denominator, oldWeight, weight);
        for (uint f = lane; f < d; f += width)
            numerator[f/width] = fma(weight, v[j*d+f], numerator[f/width] * oldWeight);
        high = next;
    }
    for (uint f = lane; f < d; f += width) out[row*d+f] = numerator[f/width] / denominator;
}
