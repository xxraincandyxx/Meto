#include <metal_stdlib>
using namespace metal;

kernel void hash_words(device const uint* words, device uint* result,
                       constant uint& rounds, uint i [[thread_position_in_grid]]) {
    uint word = words[i];
    // Each word is independent. Keep all rounds in the same thread: a new
    // dispatch per round would repeatedly write and reread the whole vector.
    for (uint r = 0; r < rounds; ++r) {
        uint hash = 2166136261u;
        for (uint shift = 0; shift < 32; shift += 8) {
            uint byte = (word >> shift) & 255u;
            hash = (hash ^ byte) * 16777619u;  // uint wraps modulo 2^32.
        }
        word = hash;
    }
    result[i] = word;
}
