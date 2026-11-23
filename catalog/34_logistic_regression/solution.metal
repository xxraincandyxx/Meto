#include <metal_stdlib>
using namespace metal;

kernel void small_newton(device const float* X, device const float* y, device float* output,
                         constant uint& n, constant uint& d, uint t [[thread_index_in_threadgroup]]) {
    // The small-problem path keeps a complete Newton iteration inside one
    // group. Threads share the probabilities, gradient and curvature without
    // copying a convergence flag back to the CPU after every iteration.
    threadgroup float residual[256], weight[256], gradient[8], beta[8], hessian[64];
    threadgroup uint converged;
    if (t < d) beta[t] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < 1000; ++step) {
        for (uint sample = t; sample < n; sample += 64) {
            float z = 0;
            for (uint f = 0; f < d; ++f) z = fma(X[sample * d + f], beta[f], z);
            // Compute the smaller sigmoid tail directly. This preserves the
            // residual of a correctly classified, high-confidence sample.
            float odds = exp(-abs(z));
            float tail = odds / (1 + odds);
            residual[sample] = z >= 0 ? (1 - y[sample]) - tail : tail - y[sample];
            weight[sample] = max(1e-8f, tail * (1 - tail));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < d * d) {
            uint i = t / d, j = t % d;
            float dot = 0;
            for (uint r = 0; r < n; ++r)
                dot = fma(X[r*d+i], weight[r] * X[r*d+j], dot);
            hessian[t] = dot + (i == j ? 1e-6f : 0);
        }
        if (t < d) {
            float dot = 0;
            for (uint r = 0; r < n; ++r) dot = fma(X[r*d+t], residual[r], dot);
            gradient[t] = dot + 1e-6f * beta[t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            // The regularized curvature is positive definite. Eliminate
            // below the diagonal and solve backward for the Newton update.
            for (uint k = 0; k < d; ++k)
                for (uint i = k + 1; i < d; ++i) {
                    float ratio = hessian[i*d+k] / hessian[k*d+k];
                    for (uint j = k + 1; j < d; ++j) hessian[i*d+j] -= ratio * hessian[k*d+j];
                    gradient[i] -= ratio * gradient[k];
                }
            float change = 0;
            for (int i = int(d) - 1; i >= 0; --i) {
                float delta = gradient[i];
                for (uint j = uint(i) + 1; j < d; ++j) delta -= hessian[i*d+j] * gradient[j];
                gradient[i] = delta / hessian[i*d+i];
                float updated = beta[i] - gradient[i];
                change += (updated - beta[i]) * (updated - beta[i]);
                beta[i] = updated;
            }
            converged = uint(change < 1e-16f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (converged) break;
    }
    if (t < d) output[t] = beta[t];
}
