#include <metal_stdlib>
using namespace metal;

inline float2 norm2(float2 x) { return x * rsqrt(0.5f * dot(x,x) + 1e-6f); }
inline float2 rotate(float2 v, float c, float s) {
    return float2(v.x * c - v.y * s, v.x * s + v.y * c);
}
inline float silu(float x) { return x / (1 + exp(-x)); }

kernel void solve(
    device const int* prompts,
    device const float* w,
    device float* logits,
    constant float& attnScale,
    uint i [[thread_position_in_grid]]) {
    // The prediction uses only the last query. Earlier tokens contribute keys
    // and values derived solely from their own digit embeddings. Cache those
    // two numbers per position and append only the freshly predicted token.
    float2 embed[10];
    for (uint digit = 0; digit < 10; ++digit)
        embed[digit] = float2(w[0] - w[1] * digit * digit, -float(digit));
    float2 keys[42];
    float vals[42];
    int token = 0;
    for (uint pos = 0; pos < 41; ++pos) {
        if (pos < 31) token = prompts[i * 31 + pos];
        float2 hidden = embed[token];
        float2 z = norm2(hidden);
        float angle = pos * (2.0f * M_PI_F / 19.0f);
        float c = cos(angle), s = sin(angle);
        keys[pos] = rotate(norm2(float2(z.x, 0.0f)), c, s);
        vals[pos] = z.y * w[4];
        if (pos < 30) continue;
        float2 q = rotate(norm2(z.x * float2(w[2], w[3])), c, s);
        // Online softmax rescales the running numerator and denominator whenever
        // a larger score arrives. It never materializes a 41-by-41 score matrix.
        float largest = -INFINITY, mass = 0, weighted = 0;
        for (uint t = 0; t <= pos; ++t) {
            float logit = dot(q, keys[t]) * attnScale;
            float next = max(largest, logit);
            float old = exp(largest - next), weight = exp(logit - next);
            weighted = weighted * old + weight * vals[t];
            mass = mass * old + weight;
            largest = next;
        }
        hidden.y += weighted / mass;
        z = norm2(hidden);
        float gate0 = dot(z, float2(w[5], w[6]));
        float gate1 = dot(z, float2(w[5] - w[6] / 1000.0f, w[6]));
        hidden.y += w[7] * (z.x * silu(gate1) - z.x * silu(gate0));
        z = norm2(hidden) * float2(w[8], w[9]);
        float best = -INFINITY;
        for (uint digit = 0; digit < 10; ++digit) {
            float score = dot(z, embed[digit]);
            logits[(i * 11 + pos - 30) * 10 + digit] = score;
            if (score > best) { best = score; token = digit; }
        }
    }
}
