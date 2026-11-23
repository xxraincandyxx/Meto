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

// When a full score row fits in threadgroup memory, four SIMD groups share a
// query. This replaces a serial update of the softmax state at every key with
// parallel dot products and one normalization. Scratch usage is independent of M.
kernel void attendShort(device const float *q, device const float *k, device const float *v,
                        device float *out, constant uint &n, constant uint &d,
                        uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint width [[threads_per_simdgroup]]) {
    threadgroup float scores[256], subtotals[4], values[4][128];
    float query[8];
    for (uint f = lane; f < d; f += width) query[f/width] = q[row*d+f];
    float peak = -INFINITY;
    for (uint j = sg; j < n; j += 4) {
        float partial = 0;
        for (uint f = lane; f < d; f += width) partial = fma(query[f/width], k[j*d+f], partial);
        float score = simd_sum(partial) * rsqrt(float(d));
        if (!lane) scores[j] = score;
        peak = max(peak, score);
    }
    if (!lane) subtotals[sg] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peak = max(max(subtotals[0], subtotals[1]), max(subtotals[2], subtotals[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mass = 0;
    for (uint j = tid; j < n; j += 4*width) {
        float weight = exp(scores[j]-peak);
        scores[j] = weight;
        mass += weight;
    }
    mass = simd_sum(mass);
    if (!lane) subtotals[sg] = mass;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Split keys four ways. Lanes within each SIMD group read neighboring value
    // features, and each group deposits one partial weighted vector in scratch.
    for (uint f = lane; f < d; f += width) {
        float sum = 0;
        for (uint j = sg; j < n; j += 4) sum = fma(scores[j], v[j*d+f], sum);
        values[sg][f] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mass = subtotals[0] + subtotals[1] + subtotals[2] + subtotals[3];
    for (uint f = tid; f < d; f += 4*width)
        out[row*d+f] = (values[0][f]+values[1][f]+values[2][f]+values[3][f]) / mass;
}

#include <metal_simdgroup_matrix>

// Eight queries share a block of keys and values. For aligned shapes the two
// dense products of attention can use the same 8x8 instructions as lesson 02.
// Scores stay in this group's 8 KiB tile through the intervening row softmax.
kernel void attendFragments(device const float *q, device const float *k, device const float *v,
                            device float *out, constant uint &n, constant uint &d,
                            uint block [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
    threadgroup float scores[8][256];
    uint query = block*8;
    for (uint key = sg*8; key < n; key += 4*8) {
        simdgroup_float8x8 accum = make_filled_simdgroup_matrix<float, 8, 8>(0);
        for (uint f = 0; f < d; f += 8) {
            simdgroup_float8x8 queryTile, keyTile;
            simdgroup_load(queryTile, q + query*d + f, d);
            // Transposing the key fragment turns eight key rows into eight
            // columns for the dot products. The original K buffer stays intact.
            simdgroup_load(keyTile, k + key*d + f, d, ulong2(0), true);
            simdgroup_multiply_accumulate(accum, queryTile, keyTile, accum);
        }
        simdgroup_store(accum, &scores[0][key], 256);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = sg; r < 8; r += 4) {
        float largest = -INFINITY;
        for (uint j = lane; j < n; j += width) largest = max(largest, scores[r][j]);
        largest = simd_max(largest);
        float sum = 0;
        for (uint j = lane; j < n; j += width) {
            float w = exp((scores[r][j]-largest) * rsqrt(float(d)));
            scores[r][j] = w;
            sum += w;
        }
        float inv = 1.f / simd_sum(sum);
        for (uint j = lane; j < n; j += width) scores[r][j] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // P[8,n] times V[n,d] yields eight complete output rows. Each group writes
    // disjoint eight-column fragments, eliminating cross-group synchronization.
    for (uint f = sg*8; f < d; f += 32) {
        simdgroup_float8x8 accum = make_filled_simdgroup_matrix<float, 8, 8>(0);
        for (uint j = 0; j < n; j += 8) {
            simdgroup_float8x8 probabilities, valueTile;
            simdgroup_load(probabilities, &scores[0][j], 256);
            simdgroup_load(valueTile, v + j*d + f, d);
            simdgroup_multiply_accumulate(accum, probabilities, valueTile, accum);
        }
        simdgroup_store(accum, out + query*d + f, d);
    }
}
