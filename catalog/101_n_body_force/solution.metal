#include <metal_stdlib>
using namespace metal;

kernel void solve(device const packed_float3* position, device const float* mass,
                  device packed_float3* force, constant uint& count,
                  uint target [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float3 here = float3(position[target]), sum = 0;
    // Share one target across the SIMD group. Neighboring lanes now stream
    // adjacent source particles and accumulate independent partial forces.
    for (uint j = lane; j < count; j += 32) {
        float3 delta = float3(position[j]) - here;
        // Squared softening keeps coincident particles finite. Cubing the
        // reciprocal distance implements the inverse-square radial force
        // because delta already has magnitude equal to the distance.
        float inv = rsqrt(dot(delta, delta) + 1e-6f);
        sum += delta * (mass[j] * inv * inv * inv);
    }
    float3 combined = float3(simd_sum(sum.x), simd_sum(sum.y), simd_sum(sum.z));
    if (lane == 0) force[target] = packed_float3(combined);
}
