# Fusing training losses on an Apple M3

Four lessons illustrate when a short Metal kernel can save repeated trips to memory. Each comparison starts from the same FP32 tensors on the GPU. The baseline runs the pinned upstream loss formula through PyTorch's eager MPS backend. The course solution computes the same result with a handwritten Metal shader.

## Results

Measured on the 8-core Apple M3 GPU with macOS 27.0.1 and PyTorch 2.15.0. Times are median synchronized milliseconds per invocation.

| Lesson | Input shape | Eager MPS | Fused solution | Speedup |
| --- | --- | ---: | ---: | ---: |
| 107 · PPO clipped surrogate | 256 × 16,384 tokens | 3.7237 ms | 1.0283 ms | 3.62× |
| 108 · DPO sequence loss | 65,536 pairs | 0.2700 ms | 0.2393 ms | 1.13× |
| 109 · GRPO surrogate | 64 × 16 × 4,096 tokens | 10.1455 ms | 1.3846 ms | 7.33× |
| 150 · Distillation KL | 256 × 32,000 logits | 11.4332 ms | 2.1875 ms | 5.23× |

Lessons 107, 108, and 109 use their upstream performance shapes. Lesson 150 uses 256 vocabulary rows of length 32,000 so the eager intermediates fit comfortably in memory. All inputs, fixtures, and dispatches stay local.

## What changed

In **107 (PPO)**, each thread reads the two log probabilities and the advantage for a token. It forms the policy ratio, clips it according to the sign of the advantage, and adds the loss to a running sum. For **108 (DPO)** the thread computes the preference margin and a numerically stable softplus. For **109 (GRPO)** the host first schedules a small reduction across each reward group. The shader reuses the resulting advantage while evaluating each token's clipping and KL penalty.

The three kernels write one partial sum per 1,024 input items. Each partial covers four consecutive stripes of 256 elements. Neighboring SIMD lanes therefore read neighboring tokens on every pass through the loop. A second reduction combines the partials across threadgroups. Inspect the group sum in [the PPO shader](../catalog/107_ppo_clipped_surrogate_loss/solution.metal).

In **150 (distillation KL)**, one threadgroup owns a vocabulary row. Its first pass finds two maxima to keep exponentials in range. Its second pass computes both normalizing constants and the teacher-weighted logit difference. These three numbers determine the row's divergence. Only the final scalar leaves the group. See [the KL shader](../catalog/150_distillation_kl_divergence/solution.metal).

## Repeat the measurements

From the repository root after `./metal-lab setup`, run the following in the VS Code terminal.

```sh
.build/catalog-env/bin/python Tools/compare_loss_fusion.py
```

The script checks both implementations against the CPU oracle, warms each three times, alternates their order, and records 15 invocations per implementation. Timings include Python dispatch and GPU synchronization. They exclude compilation and host-to-device copies. Repeating the comparison on your Mac shows how memory bandwidth, dispatch overhead, and the chosen shapes affect the payoff.
