# From a vector sum to attention

Start each lab with `./meto show ID` and `./meto practice ID`. The practice file starts with the baseline's argument list and launch geometry. For experiments that change how many values a thread owns, change the matching dispatch in the lesson's Swift file as well. Keep the baseline available for comparison.

### Follow one dispatch from Swift to the GPU

1. In `VectorAdd.swift`, the host creates two arrays and computes the expected sums. `GPU.upload` copies the inputs into shared Metal buffers. A third buffer holds the result.
2. `Program` compiles the shader source and builds a compute pipeline from the function named `add`. A pipeline is the executable GPU function plus the information Metal needs to launch it.
3. The encoder binds the input and output buffers to numbered argument slots, followed by the length. In the shader, `device` marks the address space for those buffers. `constant` marks read-only arguments shared by the grid.
4. Swift dispatches enough groups of 256 threads to cover the input. Metal gives each thread its own `thread_position_in_grid`. The addition kernel reads two elements and writes one element at the index assigned to that thread.
5. The command buffer submits the encoded work. After completion, Swift compares each output to the expected value and reports the first mismatched index. GPU timestamps on that command buffer supply the native benchmark.

[Swift runtime and buffer bindings](../Sources/MetalRuntime.swift) · [Validation and timing](../Sources/main.swift)

### 01 · Which element does a thread own?

Start with five elements and a threadgroup of 256. Metal launches the full group. Threads 0–4 produce the five sums, and the other threads skip the write. Try tracing the same input through the optimized kernel. Thread 0 writes the first four values as a `float4`. Thread 1 handles the tail. At larger sizes the two reads and one write per element dominate the arithmetic. [Baseline](../lessons/01-vector-add/baseline.metal) · [Solution](../lessons/01-vector-add/solution.metal) · [Swift launch](../lessons/01-vector-add/VectorAdd.swift)

### 02 · How often is a matrix element read?

Draw a 2×3 matrix beside a 3×4 matrix. One output element needs three multiply-adds. Its neighbor reuses the same row of the left input. The tiled kernel cooperatively loads rectangles into threadgroup memory and accumulates several outputs per thread. For aligned dimensions a SIMD group can multiply 8×8 matrix fragments directly. Read the guarded tile path first, then find the condition in Swift that selects the fragment path. [Baseline](../lessons/02-matrix-multiplication/baseline.metal) · [Solution](../lessons/02-matrix-multiplication/solution.metal) · [Swift launch](../lessons/02-matrix-multiplication/MatrixMultiplication.swift)

### 03 · Which memory accesses are neighbors?

For an input with three rows and five columns, locate input row 1, column 2. Its flat index is 7. Its destination is row 2, column 1, at flat index 7 as well. For row 0, column 2 the indices diverge to 2 and 6. The optimized kernel transposes a 4×4 rectangle in registers, so a thread reads short contiguous row segments and writes short contiguous output segments. Look for the edge path that protects odd dimensions. [Baseline](../lessons/03-matrix-transpose/baseline.metal) · [Solution](../lessons/03-matrix-transpose/solution.metal)

### 04 · How do many groups produce one number?

Each thread first accumulates up to four input values. A SIMD group combines its lanes using `simd_sum`. Its first lane writes the subtotal into threadgroup memory. A barrier makes all subtotals visible, and one SIMD group produces the group's result. Swift dispatches another pass on the resulting array until one value remains. Trace 1025 elements to see why a second pass is necessary. [Baseline](../lessons/04-reduction/baseline.metal) · [Solution](../lessons/04-reduction/solution.metal) · [Swift passes](../lessons/04-reduction/Reduction.swift)

### 05 · Can two softmax denominators be added?

For logits `[1000, 1001]`, subtract 1001 before taking exponentials. The relative weights stay the same and both exponentials fit in a float. A tile records a pair `(maximum, sum)`. To combine tiles, first express each sum relative to the largest maximum. A final parallel pass multiplies each shifted exponential by the reciprocal of the combined sum. Try equal logits. Every output should be `1/N`. [Baseline](../lessons/05-softmax/baseline.metal) · [Solution](../lessons/05-softmax/solution.metal)

### 06 · Can attention avoid storing the score matrix?

Each query needs a weighted average of the value rows. The baseline materializes one dot product per query/key pair. For an aligned short sequence, the optimized kernel computes eight score rows with matrix fragments, normalizes them in threadgroup memory, and multiplies by V. A second short-row path handles ragged shapes. For a long sequence, the streaming kernel carries a running maximum, denominator, and weighted numerator for one query. When a new maximum arrives, it rescales earlier contributions. This reuses softmax's change-of-scale calculation and removes the intermediate score matrix. For a single key, its value row must be the answer for every query. [Baseline](../lessons/06-attention/baseline.metal) · [Solution](../lessons/06-attention/solution.metal)

## Measurements on the course Mac

The 8-core M3 ran each shape twice to warm up, then the runner took the median GPU command-buffer duration across 15 samples. These values come from one complete pass of `./meto bench all`. Compare changes on the same device and shape. Vector addition gained 1.06× on this run. Vectorization still requires the same two reads and one write per element.

| ID | Shape | Baseline (ms) | Solution (ms) | Baseline / solution |
|---:|---|---:|---:|---:|
| 01 | 4194307 | 0.9055 | 0.8557 | 1.06× |
| 02 | 384, 384, 384 | 0.9569 | 0.1142 | 8.38× |
| 03 | 2049, 2051 | 0.7595 | 0.4359 | 1.74× |
| 04 | 4194304 | 0.8308 | 0.2080 | 3.99× |
| 05 | 500000 | 1.4619 | 0.0474 | 30.83× |
| 06 | 512, 256, 64 | 0.4691 | 0.0959 | 4.89× |
