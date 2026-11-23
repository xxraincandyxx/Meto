# Sampling, recurrence, and particle forces on the M3

Three upstream performance cases exercise different bottlenecks. Top-p sampling sorts the vocabulary and scans its probabilities. The recurrence carries state across time. Particle forces require every target to visit every source.

| Lesson | Upstream workload | Median latency |
| --- | --- | ---: |
| 60 · Top-p sampling | 50,000 logits, nucleus probability 0.9 | 0.502 ms |
| 82 · Linear recurrence | 64 sequences × 16,384 steps | 0.951 ms |
| 101 · N-body force | 8,192 particles, 67,108,864 interactions | 2.191 ms |

Measurements used the 8-core Apple M3 GPU on macOS 27.0.1 with PyTorch 2.15.0. The runner checks each result against the CPU oracle, runs two warm-ups, then reports the median of 15 calls. It synchronizes the GPU around each timed call, so the numbers include Python dispatch, temporary allocation inside the solution, and GPU execution. Input generation and shader compilation occur before timing.

Run the same inputs from the VS Code terminal:

```sh
./metal-lab bench 60 --performance
./metal-lab bench 82 --performance
./metal-lab bench 101 --performance
```

## Two changes tested on the same GPU

**Linear recurrence.** One lane walking a whole sequence took 2.465 ms. A hierarchical scan took 0.900 ms in the same comparison, a 2.74× speedup. Each time step represents an affine map. Combining neighboring maps first produces independent 32-step tiles, then a much smaller scan joins their results. The short-row kernel also handles the last recursion level. See [lesson 82](../catalog/82_linear_recurrence/solution.metal).

**Particle forces.** A group sharing a 128-particle tile across targets took 2.137 ms. Giving each SIMD group one target and splitting source particles across its 32 lanes took 1.959 ms, a 1.09× speedup. This mapping removes the threadgroup barriers and uses a SIMD sum at the end. An earlier comparison yielded 2.204 and 2.105 ms, respectively. The course uses the SIMD version. See [lesson 101](../catalog/101_n_body_force/solution.metal).

The paired runs reused buffers, checked each variant against the same CPU result, and interleaved 15 samples per variant after two warm-ups. The repeat for particle forces also alternated which variant ran first. The standalone medians in the table come from independent invocations. The earlier serial scan and tiled particle kernel are available in local Git history at `500b3e3` and `9b1dbbb`, respectively.
