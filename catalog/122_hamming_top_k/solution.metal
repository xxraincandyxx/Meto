#include <metal_stdlib>
using namespace metal;

kernel void distance(device const uint *codes, device const uint *query,
                     device int *out, uint row [[thread_position_in_grid]]) {
    // A code is 256 bits. XOR leaves ones exactly where two codes differ;
    // Metal's popcount counts all 32 bits of a word, including its sign bit.
    int bits = 0;
    for (uint word = 0; word < 8; ++word)
        bits += popcount(codes[8 * row + word] ^ query[word]);
    out[row] = bits;
}
