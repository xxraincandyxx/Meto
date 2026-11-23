#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a,
    device const float* x,
    device float* out,
    constant uint& length,
    uint i [[thread_position_in_grid]]) {
    // A sequence has a true time dependency. Keep its accumulator in one
    // register and place independent sequences on different grid lanes. This
    // trades time-axis parallelism for a single dispatch and O(1) scratch.
    uint start = i * length;
    float state = 0;
    for (uint t = 0; t < length; ++t) {
        state = fma(a[start + t], state, x[start + t]);
        out[start + t] = state;
    }
}

// Every step is an affine map h ↦ a*h + x. If left runs before right,
// their composition is (right.a * left.a, right.a * left.x + right.x).
// Composition is associative, so a SIMD group scans 32 steps in five rounds.
kernel void tiles(device const float* a, device const float* x, device float* h,
                  device float* prefixGain, device float* blockGain, device float* blockBias,
                  constant uint& length, constant uint& numBlocks, uint i [[thread_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]]) {
    uint block = i / 32, row = block / numBlocks;
    uint t = (block % numBlocks) * 32 + lane, pos = row * length + t;
    // Pad the last tile with identity maps so lane 31 always holds its summary.
    float gain = t < length ? a[pos] : 1, bias = t < length ? x[pos] : 0;
    for (uint offset = 1; offset < 32; offset *= 2) {
        float leftGain = simd_shuffle_up(gain, offset);
        float leftBias = simd_shuffle_up(bias, offset);
        if (lane >= offset) {
            bias = fma(gain, leftBias, bias);
            gain *= leftGain;
        }
    }
    if (t < length) { h[pos] = bias; prefixGain[pos] = gain; }
    if (lane == 31) { blockGain[block] = gain; blockBias[block] = bias; }
}

kernel void propagate(device float* h, device const float* prefixGain,
                      device const float* ends, constant uint& length,
                      constant uint& numBlocks, uint i [[thread_position_in_grid]]) {
    uint block = (i % length) / 32;
    // Carry the preceding tile's final state through each local prefix map.
    if (block) h[i] = fma(prefixGain[i], ends[(i / length) * numBlocks + block - 1], h[i]);
}
