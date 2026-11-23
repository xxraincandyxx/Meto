#include <metal_stdlib>
using namespace metal;

// Materialize Q times K-transpose. Linear index i denotes a query/key pair.
kernel void score(device const float *q, device const float *k, device float *scores,
                  constant uint &m, constant uint &n, constant uint &d,
                  uint i [[thread_position_in_grid]]) {
    if (i >= m*n) return;
    float dot = 0;
    for (uint f = 0; f < d; ++f) dot = fma(q[(i/n)*d+f], k[(i%n)*d+f], dot);
    scores[i] = dot * rsqrt(float(d));
}

// One output feature per thread. All features of a query use the same softmax
// probabilities; the optimized kernel takes advantage of this repeated work.
kernel void mixValues(device const float *scores, device const float *v, device float *out,
                      constant uint &m, constant uint &n, constant uint &d,
                      uint i [[thread_position_in_grid]]) {
    if (i >= m*d) return;
    device const float *row = scores + (i/d)*n;
    float high = -INFINITY;
    for (uint j = 0; j < n; ++j) high = max(high, row[j]);
    float total = 0, weighted = 0;
    for (uint j = 0; j < n; ++j) {
        float w = exp(row[j] - high);
        total += w;
        weighted = fma(w, v[j*d + i%d], weighted);
    }
    out[i] = weighted / total;
}
