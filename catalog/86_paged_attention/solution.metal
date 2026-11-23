#include <metal_stdlib>
using namespace metal;

inline uint cacheOffset(device const int* pages, uint b, uint t, uint head,
                    uint heads, uint dim, uint blockSize, uint maxPages) {
    uint page = uint(pages[b * maxPages + t / blockSize]);
    return ((page * blockSize + t % blockSize) * heads + head) * dim;
}

kernel void score(device const float* q, device const float* keys, device const int* pages,
                  device const int* lengths, device float* scores, constant uint& heads,
                  constant uint& dim, constant uint& blockSize, constant uint& maxPages,
                  uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint tokenRow = i / 32, length = maxPages * blockSize;
    uint row = tokenRow / length, t = tokenRow % length;
    uint b = row / heads, head = row % heads;
    if (t >= uint(lengths[b])) {
        if (lane == 0) scores[tokenRow] = -INFINITY;
        return;
    }
    uint base = cacheOffset(pages, b, t, head, heads, dim, blockSize, maxPages);
    float dotProduct = 0;
    for (uint d = lane; d < dim; d += 32)
        dotProduct = fma(q[row * dim + d], keys[base + d], dotProduct);
    // Shuffle-and-add across lanes combines the partial dot products.
    float logit = simd_sum(dotProduct) * rsqrt(float(dim));
    if (lane == 0) scores[tokenRow] = logit;
}

kernel void mix(device const float* weights, device const float* values, device const int* pages,
                device const int* lengths, device float* out, constant uint& heads,
                constant uint& dim, constant uint& blockSize, constant uint& maxPages,
                uint i [[thread_position_in_grid]]) {
    uint row = i / dim, b = row / heads, head = row % heads;
    float sum = 0;
    for (uint t = 0; t < uint(lengths[b]); ++t) {
        uint base = cacheOffset(pages, b, t, head, heads, dim, blockSize, maxPages);
        sum = fma(weights[row * (maxPages * blockSize) + t], values[base + i % dim], sum);
    }
    out[i] = sum;
}
