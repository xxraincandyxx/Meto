# Fundamentals on the Apple M3

The table measures one local invocation of each solution, including Python dispatch and waiting for Metal to finish. The inputs were already on the GPU. Large working sets helped distinguish memory traffic from the fixed cost of launching a small kernel.

This run used an eight-core Apple M3 GPU, macOS 27.0.1, Python 3.13.12 and PyTorch 2.15.0. Each workload passed a comparison against the pinned CPU reference at the challenge's tolerance. The runner restored mutable inputs, performed two warmups and took the median of fifteen synchronized trials. Compilation, test generation and CPU grading took place outside the timed interval.

## Reproduce from VS Code

After running `./metal-lab setup`, use the integrated terminal. The first command runs all 44 bounded workloads. Pass individual IDs to shorten a run. The second command times the alternative implementations listed below.

```sh
.build/catalog-env/bin/python Tools/benchmark_fundamentals.py
.build/catalog-env/bin/python Tools/benchmark_fundamentals.py --compare
```

## Measured implementation choices

| Lesson | Default | Alternative | Default ms | Alternative ms |
|---|---|---|---:|---:|
| 15 | Float sort on MPS | Encode ordered integers and radix-sort them | 2.045 | 3.538 |
| 25 | SIMD log-sum-exp per row | Backend cross-entropy with converted labels | 0.293 | 0.480 |
| 27 | Fused partial squared-error reduction | Backend mean-squared-error loss | 0.317 | 0.460 |
| 34 | Single-group Newton solve | Host-dispatched Newton iterations | 0.990 | 494.067 |
| 39 | FFT on complex views | Pack and unpack complex tensors | 0.245 | 0.445 |
| 41 | GEMM writes the supplied output | Call the linear module and copy its return | 0.239 | 0.259 |
| 46 | Single-group frontier and visitation bits | Device-wide queues with synchronized frontiers | 0.552 | 28.178 |

In lesson 34, the small feature matrix and its gradient fit in threadgroup memory. The group can run Newton iterations without returning to Python. Computing the smaller sigmoid tail directly also preserves small residuals when predictions approach one. Larger problems use device-wide linear algebra. In lesson 46, a queue of 4096 sixteen-bit cell indices and 128 visitation words covers the measured grid in one group. Larger grids use separate frontiers in device memory.

The radix experiment for lesson 15 adds two conversions and eight pairs of digit-counting and scattering passes. The MPS float sort completed sooner on this workload. The candidate remains available for comparing the trade-off on another GPU. The timings around a few tenths of a millisecond include substantial dispatch overhead and can shift with device load.

## All bounded workloads

| ID | Exercise | Workload | Median ms |
|---:|---|---|---:|
| 7 | Color inversion | 512×512 RGBA | 0.403 |
| 8 | Matrix addition | 512×512 | 0.260 |
| 9 | 1d convolution | 65536, filter 257 | 0.883 |
| 10 | 2d convolution | 256×256, filter 7×7 | 0.385 |
| 11 | 3d convolution | 32³, filter 3³ | 0.291 |
| 12 | Multi head attention | 256 tokens, 8 heads × 32 | 0.419 |
| 13 | Histogramming | 262144, 256 bins | 0.234 |
| 14 | Multi agent sim | 1024 agents | 0.595 |
| 15 | Sorting | 262144 | 2.045 |
| 16 | Prefix sum | 262144 | 0.259 |
| 17 | Dot product | 1048576 | 0.339 |
| 18 | Sparse matrix vector multiplication | 256×512, ~65% zero | 0.205 |
| 19 | Reverse array | 262144 | 0.217 |
| 20 | Kmeans clustering | 2048 points, k=5, 8 steps | 0.669 |
| 21 | Relu | 262144 | 0.210 |
| 22 | Gemm | 256³, FP16 | 0.266 |
| 23 | Leaky relu | 262144 | 0.195 |
| 24 | Rainbow table | 262144, 20 rounds | 0.281 |
| 25 | Categorical cross entropy loss | 2048×512 | 0.293 |
| 26 | Multi head cross attention | 128 queries, 256 keys, 4 heads × 32 | 0.218 |
| 27 | Mean squared error | 1048576 | 0.317 |
| 28 | Gaussian blur | 256×256, filter 7×7 | 0.237 |
| 29 | Top k selection | 262144, top 32 | 0.850 |
| 30 | Batched matrix multiplication | 16 batches, 64³ | 0.205 |
| 31 | Matrix copy | 512×512 | 0.195 |
| 32 | Int8 quantized matmul | 256³, INT8 | 0.286 |
| 33 | Ordinary least squares | 1024 samples, 16 features | 0.367 |
| 34 | Logistic regression | 16 samples, 8 features | 0.990 |
| 35 | Monte carlo integration | 262144 | 0.233 |
| 36 | Radix sort | 262144 unsigned keys | 1.631 |
| 37 | Matrix power | 128×128, power 7 | 0.304 |
| 38 | Nearest neighbor | 1024 XYZ points | 0.335 |
| 39 | Fast fourier transform | 65536 complex samples | 0.245 |
| 40 | Batch normalization | 1024×128 | 0.343 |
| 41 | Simple inference | 128 samples, 256→128 | 0.239 |
| 42 | 2d max pooling | 4×8×64×64, 3×3 stride 2 | 0.282 |
| 43 | Count array element | 262144 | 0.208 |
| 44 | Count 2d array element | 512×512 | 0.205 |
| 45 | Count 3d array element | 64×64×64 | 0.213 |
| 46 | Bfs shortest path | 64×64, open grid | 0.552 |
| 47 | Subarray sum | 262144, middle half | 0.212 |
| 48 | 2d subarray sum | 512², middle 256² | 0.217 |
| 49 | 3d subarray sum | 64³, middle 32³ | 0.216 |
| 50 | Rms normalization | 65536 | 0.240 |
