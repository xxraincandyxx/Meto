# Curriculum

Read a prompt from the integrated terminal with `./meto show ID`. Follow the solution link to see the dispatch and the algorithm. A shader link opens the Metal Shading Language code used by that lesson. IDs match the pinned reference.

The last column records one verified run on an 8-core Apple M3 using the default local benchmark shape for each lesson. Lessons 1–6 report GPU command-buffer time. The remaining lessons include host dispatch and synchronization. Reproduce a row with `./meto bench ID` to see its exact shape, or time the course with `./meto bench all`.

## 01–06 · Native Metal foundations

| ID | Problem | Level | Implementation | M3 median (ms) |
|---:|---|---|---|---:|
| 01 | [Vector Addition](../reference/leetgpu-challenges/challenges/easy/1_vector_add/challenge.html) | easy | [Baseline](../lessons/01-vector-add/baseline.metal) · [Shader](../lessons/01-vector-add/solution.metal) | 0.8557 |
| 02 | [Matrix Multiplication](../reference/leetgpu-challenges/challenges/easy/2_matrix_multiplication/challenge.html) | easy | [Baseline](../lessons/02-matrix-multiplication/baseline.metal) · [Shader](../lessons/02-matrix-multiplication/solution.metal) | 0.1142 |
| 03 | [Matrix Transpose](../reference/leetgpu-challenges/challenges/easy/3_matrix_transpose/challenge.html) | easy | [Baseline](../lessons/03-matrix-transpose/baseline.metal) · [Shader](../lessons/03-matrix-transpose/solution.metal) | 0.4359 |
| 04 | [Reduction](../reference/leetgpu-challenges/challenges/medium/4_reduction/challenge.html) | medium | [Baseline](../lessons/04-reduction/baseline.metal) · [Shader](../lessons/04-reduction/solution.metal) | 0.2080 |
| 05 | [Softmax](../reference/leetgpu-challenges/challenges/medium/5_softmax/challenge.html) | medium | [Baseline](../lessons/05-softmax/baseline.metal) · [Shader](../lessons/05-softmax/solution.metal) | 0.0474 |
| 06 | [Softmax Attention](../reference/leetgpu-challenges/challenges/medium/6_softmax_attention/challenge.html) | medium | [Baseline](../lessons/06-attention/baseline.metal) · [Shader](../lessons/06-attention/solution.metal) | 0.0959 |

## 07–25 · Images, data movement, and parallel primitives

| ID | Problem | Level | Implementation | M3 median (ms) |
|---:|---|---|---|---:|
| 07 | [Color Inversion](../reference/leetgpu-challenges/challenges/easy/7_color_inversion/challenge.html) | easy | [Solution](../catalog/7_color_inversion/solution.py) · [Shader](../catalog/7_color_inversion/solution.metal) | 0.3282 |
| 08 | [Matrix Addition](../reference/leetgpu-challenges/challenges/easy/8_matrix_addition/challenge.html) | easy | [Solution](../catalog/8_matrix_addition/solution.py) · [Shader](../catalog/8_matrix_addition/solution.metal) | 0.1903 |
| 09 | [1D Convolution](../reference/leetgpu-challenges/challenges/easy/9_1d_convolution/challenge.html) | easy | [Solution](../catalog/9_1d_convolution/solution.py) | 0.2138 |
| 10 | [2D Convolution](../reference/leetgpu-challenges/challenges/medium/10_2d_convolution/challenge.html) | medium | [Solution](../catalog/10_2d_convolution/solution.py) | 0.2826 |
| 11 | [3D Convolution](../reference/leetgpu-challenges/challenges/medium/11_3d_convolution/challenge.html) | medium | [Solution](../catalog/11_3d_convolution/solution.py) · [Shader](../catalog/11_3d_convolution/solution.metal) | 0.2280 |
| 12 | [Multi-Head Attention](../reference/leetgpu-challenges/challenges/hard/12_multi_head_attention/challenge.html) | hard | [Solution](../catalog/12_multi_head_attention/solution.py) | 0.3512 |
| 13 | [Histogramming](../reference/leetgpu-challenges/challenges/medium/13_histogramming/challenge.html) | medium | [Solution](../catalog/13_histogramming/solution.py) · [Shader](../catalog/13_histogramming/solution.metal) | 0.1941 |
| 14 | [Multi-Agent Simulation](../reference/leetgpu-challenges/challenges/hard/14_multi_agent_sim/challenge.html) | hard | [Solution](../catalog/14_multi_agent_sim/solution.py) · [Shader](../catalog/14_multi_agent_sim/solution.metal) | 0.4029 |
| 15 | [Sorting](../reference/leetgpu-challenges/challenges/hard/15_sorting/challenge.html) | hard | [Solution](../catalog/15_sorting/solution.py) | 0.2970 |
| 16 | [Prefix Sum](../reference/leetgpu-challenges/challenges/medium/16_prefix_sum/challenge.html) | medium | [Solution](../catalog/16_prefix_sum/solution.py) | 0.1630 |
| 17 | [Dot Product](../reference/leetgpu-challenges/challenges/medium/17_dot_product/challenge.html) | medium | [Solution](../catalog/17_dot_product/solution.py) | 0.2132 |
| 18 | [Sparse Matrix-Vector Multiplication](../reference/leetgpu-challenges/challenges/medium/18_sparse_matrix_vector_multiplication/challenge.html) | medium | [Solution](../catalog/18_sparse_matrix_vector_multiplication/solution.py) | 0.1849 |
| 19 | [Reverse Array](../reference/leetgpu-challenges/challenges/easy/19_reverse_array/challenge.html) | easy | [Solution](../catalog/19_reverse_array/solution.py) · [Shader](../catalog/19_reverse_array/solution.metal) | 0.1715 |
| 20 | [K-Means Clustering](../reference/leetgpu-challenges/challenges/hard/20_kmeans_clustering/challenge.html) | hard | [Solution](../catalog/20_kmeans_clustering/solution.py) · [Shader](../catalog/20_kmeans_clustering/solution.metal) | 1.7760 |
| 21 | [ReLU](../reference/leetgpu-challenges/challenges/easy/21_relu/challenge.html) | easy | [Solution](../catalog/21_relu/solution.py) · [Shader](../catalog/21_relu/solution.metal) | 0.1684 |
| 22 | [General Matrix Multiplication (GEMM)](../reference/leetgpu-challenges/challenges/medium/22_gemm/challenge.html) | medium | [Solution](../catalog/22_gemm/solution.py) | 0.2101 |
| 23 | [Leaky ReLU](../reference/leetgpu-challenges/challenges/easy/23_leaky_relu/challenge.html) | easy | [Solution](../catalog/23_leaky_relu/solution.py) · [Shader](../catalog/23_leaky_relu/solution.metal) | 0.1703 |
| 24 | [Rainbow Table](../reference/leetgpu-challenges/challenges/easy/24_rainbow_table/challenge.html) | easy | [Solution](../catalog/24_rainbow_table/solution.py) · [Shader](../catalog/24_rainbow_table/solution.metal) | 0.1725 |
| 25 | [Categorical Cross Entropy Loss](../reference/leetgpu-challenges/challenges/medium/25_categorical_cross_entropy_loss/challenge.html) | medium | [Solution](../catalog/25_categorical_cross_entropy_loss/solution.py) · [Shader](../catalog/25_categorical_cross_entropy_loss/solution.metal) | 0.1712 |

## 26–50 · Linear algebra, statistics, and grids

| ID | Problem | Level | Implementation | M3 median (ms) |
|---:|---|---|---|---:|
| 26 | [Multi-Head Cross-Attention](../reference/leetgpu-challenges/challenges/hard/26_multi_head_cross_attention/challenge.html) | hard | [Solution](../catalog/26_multi_head_cross_attention/solution.py) | 0.4363 |
| 27 | [Mean Squared Error](../reference/leetgpu-challenges/challenges/medium/27_mean_squared_error/challenge.html) | medium | [Solution](../catalog/27_mean_squared_error/solution.py) · [Shader](../catalog/27_mean_squared_error/solution.metal) | 0.1908 |
| 28 | [Gaussian Blur](../reference/leetgpu-challenges/challenges/medium/28_gaussian_blur/challenge.html) | medium | [Solution](../catalog/28_gaussian_blur/solution.py) | 0.2025 |
| 29 | [Top K Selection](../reference/leetgpu-challenges/challenges/medium/29_top_k_selection/challenge.html) | medium | [Solution](../catalog/29_top_k_selection/solution.py) | 8.9938 |
| 30 | [Batched Matrix Multiplication](../reference/leetgpu-challenges/challenges/medium/30_batched_matrix_multiplication/challenge.html) | medium | [Solution](../catalog/30_batched_matrix_multiplication/solution.py) | 0.1916 |
| 31 | [Matrix Copy](../reference/leetgpu-challenges/challenges/easy/31_matrix_copy/challenge.html) | easy | [Solution](../catalog/31_matrix_copy/solution.py) | 0.1610 |
| 32 | [INT8 Quantized MatMul](../reference/leetgpu-challenges/challenges/medium/32_int8_quantized_matmul/challenge.html) | medium | [Solution](../catalog/32_int8_quantized_matmul/solution.py) | 0.2255 |
| 33 | [Ordinary Least Squares](../reference/leetgpu-challenges/challenges/medium/33_ordinary_least_squares/challenge.html) | medium | [Solution](../catalog/33_ordinary_least_squares/solution.py) | 0.3180 |
| 34 | [Logistic Regression](../reference/leetgpu-challenges/challenges/medium/34_logistic_regression/challenge.html) | medium | [Solution](../catalog/34_logistic_regression/solution.py) · [Shader](../catalog/34_logistic_regression/solution.metal) | 3.9345 |
| 35 | [Monte Carlo Integration](../reference/leetgpu-challenges/challenges/medium/35_monte_carlo_integration/challenge.html) | medium | [Solution](../catalog/35_monte_carlo_integration/solution.py) | 0.1608 |
| 36 | [Radix Sort](../reference/leetgpu-challenges/challenges/hard/36_radix_sort/challenge.html) | hard | [Solution](../catalog/36_radix_sort/solution.py) · [Shader](../catalog/36_radix_sort/solution.metal) | 0.3728 |
| 37 | [Matrix Power](../reference/leetgpu-challenges/challenges/medium/37_matrix_power/challenge.html) | medium | [Solution](../catalog/37_matrix_power/solution.py) | 0.2162 |
| 38 | [Nearest Neighbor](../reference/leetgpu-challenges/challenges/medium/38_nearest_neighbor/challenge.html) | medium | [Solution](../catalog/38_nearest_neighbor/solution.py) · [Shader](../catalog/38_nearest_neighbor/solution.metal) | 0.2168 |
| 39 | [Fast Fourier Transform](../reference/leetgpu-challenges/challenges/hard/39_Fast_Fourier_transform/challenge.html) | hard | [Solution](../catalog/39_Fast_Fourier_transform/solution.py) | 0.1996 |
| 40 | [Batch Normalization](../reference/leetgpu-challenges/challenges/medium/40_batch_normalization/challenge.html) | medium | [Solution](../catalog/40_batch_normalization/solution.py) | 0.2422 |
| 41 | [Simple Inference](../reference/leetgpu-challenges/challenges/easy/41_simple_inference/challenge.html) | easy | [Solution](../catalog/41_simple_inference/solution.py) | 0.1873 |
| 42 | [2D Max Pooling](../reference/leetgpu-challenges/challenges/medium/42_2d_max_pooling/challenge.html) | medium | [Solution](../catalog/42_2d_max_pooling/solution.py) | 0.1691 |
| 43 | [Count Array Element](../reference/leetgpu-challenges/challenges/medium/43_count_array_element/challenge.html) | medium | [Solution](../catalog/43_count_array_element/solution.py) · [Shader](../catalog/43_count_array_element/solution.metal) | 0.1546 |
| 44 | [Count 2D Array Element](../reference/leetgpu-challenges/challenges/medium/44_count_2d_array_element/challenge.html) | medium | [Solution](../catalog/44_count_2d_array_element/solution.py) · [Shader](../catalog/43_count_array_element/solution.metal) | 0.2493 |
| 45 | [Count 3D Array Element](../reference/leetgpu-challenges/challenges/medium/45_count_3d_array_element/challenge.html) | medium | [Solution](../catalog/45_count_3d_array_element/solution.py) · [Shader](../catalog/43_count_array_element/solution.metal) | 0.5061 |
| 46 | [BFS Shortest Path](../reference/leetgpu-challenges/challenges/hard/46_bfs_shortest_path/challenge.html) | hard | [Solution](../catalog/46_bfs_shortest_path/solution.py) · [Shader](../catalog/46_bfs_shortest_path/solution.metal) | 0.1984 |
| 47 | [Subarray Sum](../reference/leetgpu-challenges/challenges/medium/47_subarray_sum/challenge.html) | medium | [Solution](../catalog/47_subarray_sum/solution.py) | 0.1835 |
| 48 | [2D Subarray Sum](../reference/leetgpu-challenges/challenges/medium/48_2d_subarray_sum/challenge.html) | medium | [Solution](../catalog/48_2d_subarray_sum/solution.py) | 0.2275 |
| 49 | [3D Subarray Sum](../reference/leetgpu-challenges/challenges/medium/49_3d_subarray_sum/challenge.html) | medium | [Solution](../catalog/49_3d_subarray_sum/solution.py) | 0.1700 |
| 50 | [RMS Normalization](../reference/leetgpu-challenges/challenges/medium/50_rms_normalization/challenge.html) | medium | [Solution](../catalog/50_rms_normalization/solution.py) · [Shader](../catalog/50_rms_normalization/solution.metal) | 0.1735 |

## 51–75 · Attention, scans, and sparse operations

| ID | Problem | Level | Implementation | M3 median (ms) |
|---:|---|---|---|---:|
| 51 | [Max Subarray Sum](../reference/leetgpu-challenges/challenges/medium/51_max_subarray_sum/challenge.html) | medium | [Solution](../catalog/51_max_subarray_sum/solution.py) | 0.2046 |
| 52 | [Sigmoid Linear Unit](../reference/leetgpu-challenges/challenges/easy/52_silu/challenge.html) | easy | [Solution](../catalog/52_silu/solution.py) · [Shader](../catalog/52_silu/solution.metal) | 0.1485 |
| 53 | [Causal Self-Attention](../reference/leetgpu-challenges/challenges/hard/53_casual_attention/challenge.html) | hard | [Solution](../catalog/53_casual_attention/solution.py) | 0.1895 |
| 54 | [Swish-Gated Linear Unit](../reference/leetgpu-challenges/challenges/easy/54_swiglu/challenge.html) | easy | [Solution](../catalog/54_swiglu/solution.py) · [Shader](../catalog/54_swiglu/solution.metal) | 0.1480 |
| 55 | [Attention with Linear Biases](../reference/leetgpu-challenges/challenges/medium/55_attn_w_linear_bias/challenge.html) | medium | [Solution](../catalog/55_attn_w_linear_bias/solution.py) | 0.1986 |
| 56 | [Linear Self-Attention](../reference/leetgpu-challenges/challenges/hard/56_linear_attention/challenge.html) | hard | [Solution](../catalog/56_linear_attention/solution.py) | 0.2550 |
| 57 | [FP16 Batched Matrix Multiplication](../reference/leetgpu-challenges/challenges/medium/57_fp16_batched_matmul/challenge.html) | medium | [Solution](../catalog/57_fp16_batched_matmul/solution.py) | 0.1938 |
| 58 | [FP16 Dot Product](../reference/leetgpu-challenges/challenges/medium/58_fp16_dot_product/challenge.html) | medium | [Solution](../catalog/58_fp16_dot_product/solution.py) · [Shader](../catalog/58_fp16_dot_product/solution.metal) | 0.1599 |
| 59 | [Sliding Window Self-Attention](../reference/leetgpu-challenges/challenges/hard/59_sliding_window_attn/challenge.html) | hard | [Solution](../catalog/59_sliding_window_attn/solution.py) | 0.2234 |
| 60 | [Top-p Sampling](../reference/leetgpu-challenges/challenges/medium/60_top_p_sampling/challenge.html) | medium | [Solution](../catalog/60_top_p_sampling/solution.py) · [Shader](../catalog/60_top_p_sampling/solution.metal) | 0.2490 |
| 61 | [Rotary Positional Embedding](../reference/leetgpu-challenges/challenges/medium/61_rope_embedding/challenge.html) | medium | [Solution](../catalog/61_rope_embedding/solution.py) · [Shader](../catalog/61_rope_embedding/solution.metal) | 0.1843 |
| 62 | [Value Clipping](../reference/leetgpu-challenges/challenges/easy/62_value_clipping/challenge.html) | easy | [Solution](../catalog/62_value_clipping/solution.py) · [Shader](../catalog/62_value_clipping/solution.metal) | 0.1604 |
| 63 | [Interleave Arrays](../reference/leetgpu-challenges/challenges/easy/63_interleave/challenge.html) | easy | [Solution](../catalog/63_interleave/solution.py) · [Shader](../catalog/63_interleave/solution.metal) | 0.1890 |
| 64 | [Weight Dequantization](../reference/leetgpu-challenges/challenges/medium/64_weight_dequantization/challenge.html) | medium | [Solution](../catalog/64_weight_dequantization/solution.py) · [Shader](../catalog/64_weight_dequantization/solution.metal) | 0.3560 |
| 65 | [Gaussian Error Gated Linear Unit](../reference/leetgpu-challenges/challenges/easy/65_geglu/challenge.html) | easy | [Solution](../catalog/65_geglu/solution.py) | 0.2153 |
| 66 | [RGB to Grayscale](../reference/leetgpu-challenges/challenges/easy/66_rgb_to_grayscale/challenge.html) | easy | [Solution](../catalog/66_rgb_to_grayscale/solution.py) · [Shader](../catalog/66_rgb_to_grayscale/solution.metal) | 0.1713 |
| 67 | [MoE Top-K Gating](../reference/leetgpu-challenges/challenges/medium/67_moe_topk_gating/challenge.html) | medium | [Solution](../catalog/67_moe_topk_gating/solution.py) | 0.1889 |
| 68 | [Sigmoid Activation](../reference/leetgpu-challenges/challenges/easy/68_sigmoid/challenge.html) | easy | [Solution](../catalog/68_sigmoid/solution.py) · [Shader](../catalog/68_sigmoid/solution.metal) | 0.1695 |
| 69 | [2D Jacobi Stencil](../reference/leetgpu-challenges/challenges/medium/69_jacobi_stencil_2d/challenge.html) | medium | [Solution](../catalog/69_jacobi_stencil_2d/solution.py) · [Shader](../catalog/69_jacobi_stencil_2d/solution.metal) | 0.3204 |
| 70 | [Segmented Exclusive Prefix Sum](../reference/leetgpu-challenges/challenges/medium/70_segmented_prefix_sum/challenge.html) | medium | [Solution](../catalog/70_segmented_prefix_sum/solution.py) · [Shader](../catalog/70_segmented_prefix_sum/solution.metal) | 0.2697 |
| 71 | [Parallel Merge](../reference/leetgpu-challenges/challenges/medium/71_parallel_merge/challenge.html) | medium | [Solution](../catalog/71_parallel_merge/solution.py) · [Shader](../catalog/71_parallel_merge/solution.metal) | 0.1699 |
| 72 | [Stream Compaction](../reference/leetgpu-challenges/challenges/medium/72_stream_compaction/challenge.html) | medium | [Solution](../catalog/72_stream_compaction/solution.py) · [Shader](../catalog/72_stream_compaction/solution.metal) | 0.1973 |
| 73 | [All-Pairs Shortest Paths](../reference/leetgpu-challenges/challenges/hard/73_all_pairs_shortest_paths/challenge.html) | hard | [Solution](../catalog/73_all_pairs_shortest_paths/solution.py) · [Shader](../catalog/73_all_pairs_shortest_paths/solution.metal) | 8.1985 |
| 74 | [GPT-2 Transformer Block](../reference/leetgpu-challenges/challenges/hard/74_gpt2_block/challenge.html) | hard | [Solution](../catalog/74_gpt2_block/solution.py) | 2.6152 |
| 75 | [Sparse Matrix-Dense Matrix Multiplication](../reference/leetgpu-challenges/challenges/medium/75_sparse_matrix_dense_matrix_multiplication/challenge.html) | medium | [Solution](../catalog/75_sparse_matrix_dense_matrix_multiplication/solution.py) | 0.4960 |

## 76–101 · Transformers, quantization, and routing

| ID | Problem | Level | Implementation | M3 median (ms) |
|---:|---|---|---|---:|
| 76 | [Adder Transformer Inference](../reference/leetgpu-challenges/challenges/medium/76_adder_transformer/challenge.html) | medium | [Solution](../catalog/76_adder_transformer/solution.py) · [Shader](../catalog/76_adder_transformer/solution.metal) | 1.0453 |
| 77 | [Integral Image](../reference/leetgpu-challenges/challenges/medium/77_integral_image/challenge.html) | medium | [Solution](../catalog/77_integral_image/solution.py) | 0.6158 |
| 78 | [2D FFT](../reference/leetgpu-challenges/challenges/medium/78_2d_fft/challenge.html) | medium | [Solution](../catalog/78_2d_fft/solution.py) | 0.3300 |
| 79 | [Connected Components](../reference/leetgpu-challenges/challenges/medium/79_connected_components/challenge.html) | medium | [Solution](../catalog/79_connected_components/solution.py) · [Shader](../catalog/79_connected_components/solution.metal) | 0.3117 |
| 80 | [Grouped Query Attention](../reference/leetgpu-challenges/challenges/medium/80_grouped_query_attention/challenge.html) | medium | [Solution](../catalog/80_grouped_query_attention/solution.py) | 0.3360 |
| 81 | [INT4 Weight-Only Quantized MatMul](../reference/leetgpu-challenges/challenges/medium/81_int4_matmul/challenge.html) | medium | [Solution](../catalog/81_int4_matmul/solution.py) · [Shader](../catalog/81_int4_matmul/solution.metal) | 0.3367 |
| 82 | [Linear Recurrence](../reference/leetgpu-challenges/challenges/medium/82_linear_recurrence/challenge.html) | medium | [Solution](../catalog/82_linear_recurrence/solution.py) · [Shader](../catalog/82_linear_recurrence/solution.metal) | 0.2285 |
| 83 | [Fused Residual Add and RMS Norm](../reference/leetgpu-challenges/challenges/medium/83_fused_residual_add_rms_norm/challenge.html) | medium | [Solution](../catalog/83_fused_residual_add_rms_norm/solution.py) · [Shader](../catalog/83_fused_residual_add_rms_norm/solution.metal) | 0.3099 |
| 84 | [SwiGLU MLP Block](../reference/leetgpu-challenges/challenges/medium/84_swiglu_mlp_block/challenge.html) | medium | [Solution](../catalog/84_swiglu_mlp_block/solution.py) · [Shader](../catalog/84_swiglu_mlp_block/solution.metal) | 1.0397 |
| 85 | [LoRA Linear](../reference/leetgpu-challenges/challenges/medium/85_lora_linear/challenge.html) | medium | [Solution](../catalog/85_lora_linear/solution.py) | 0.5870 |
| 86 | [Paged KV-Cache Attention](../reference/leetgpu-challenges/challenges/medium/86_paged_attention/challenge.html) | medium | [Solution](../catalog/86_paged_attention/solution.py) · [Shader](../catalog/86_paged_attention/solution.metal) | 0.5843 |
| 87 | [Speculative Decoding Verification](../reference/leetgpu-challenges/challenges/medium/87_speculative_decoding_verification/challenge.html) | medium | [Solution](../catalog/87_speculative_decoding_verification/solution.py) | 0.3652 |
| 88 | [Shared-Prefix Cascade Attention](../reference/leetgpu-challenges/challenges/medium/88_shared_prefix_attention/challenge.html) | medium | [Solution](../catalog/88_shared_prefix_attention/solution.py) | 10.1391 |
| 89 | [NVFP4 Matrix Multiplication](../reference/leetgpu-challenges/challenges/hard/89_fp4_matmul/challenge.html) | hard | [Solution](../catalog/89_fp4_matmul/solution.py) · [Shader](../catalog/89_fp4_matmul/solution.metal) | 1.0699 |
| 90 | [Causal Depthwise Conv1d](../reference/leetgpu-challenges/challenges/medium/90_causal_depthwise_conv1d/challenge.html) | medium | [Solution](../catalog/90_causal_depthwise_conv1d/solution.py) · [Shader](../catalog/90_causal_depthwise_conv1d/solution.metal) | 0.1893 |
| 91 | [MoE Token Dispatch](../reference/leetgpu-challenges/challenges/medium/91_moe_token_dispatch/challenge.html) | medium | [Solution](../catalog/91_moe_token_dispatch/solution.py) | 0.2798 |
| 92 | [Decaying Causal Attention](../reference/leetgpu-challenges/challenges/medium/92_decaying_causal_attention/challenge.html) | medium | [Solution](../catalog/92_decaying_causal_attention/solution.py) | 0.4090 |
| 93 | [Llama Transformer Block](../reference/leetgpu-challenges/challenges/hard/93_llama_transformer_block/challenge.html) | hard | [Solution](../catalog/93_llama_transformer_block/solution.py) | 1.5983 |
| 94 | [SSM Selective Scan](../reference/leetgpu-challenges/challenges/medium/94_ssm_selective_scan/challenge.html) | medium | [Solution](../catalog/94_ssm_selective_scan/solution.py) · [Shader](../catalog/94_ssm_selective_scan/solution.metal) | 0.5947 |
| 95 | [TurboQuant KV Cache Attention](../reference/leetgpu-challenges/challenges/hard/95_turboquant_attention/challenge.html) | hard | [Solution](../catalog/95_turboquant_attention/solution.py) | 0.9465 |
| 96 | [INT8 KV-Cache Attention](../reference/leetgpu-challenges/challenges/medium/96_int8_kv_cache_attention/challenge.html) | medium | [Solution](../catalog/96_int8_kv_cache_attention/solution.py) · [Shader](../catalog/96_int8_kv_cache_attention/solution.metal) | 0.3310 |
| 98 | [Beam Search Step](../reference/leetgpu-challenges/challenges/medium/98_beam_search_step/challenge.html) | medium | [Solution](../catalog/98_beam_search_step/solution.py) | 0.4499 |
| 99 | [Grouped GEMM](../reference/leetgpu-challenges/challenges/medium/99_grouped_gemm/challenge.html) | medium | [Solution](../catalog/99_grouped_gemm/solution.py) | 0.7848 |
| 101 | [N-body Gravitational Force](../reference/leetgpu-challenges/challenges/medium/101_n_body_force/challenge.html) | medium | [Solution](../catalog/101_n_body_force/solution.py) · [Shader](../catalog/101_n_body_force/solution.metal) | 0.2233 |

## 102–125 · Training, decoding, and vision

| ID | Problem | Level | Implementation | M3 median (ms) |
|---:|---|---|---|---:|
| 102 | [Variable-Length Causal Attention](../reference/leetgpu-challenges/challenges/medium/102_varlen_causal_attention/challenge.html) | medium | [Solution](../catalog/102_varlen_causal_attention/solution.py) | 0.5612 |
| 103 | [Hadamard Transform](../reference/leetgpu-challenges/challenges/medium/103_hadamard_transform/challenge.html) | medium | [Solution](../catalog/103_hadamard_transform/solution.py) · [Shader](../catalog/103_hadamard_transform/solution.metal) | 0.2990 |
| 104 | [Min-P Sampling](../reference/leetgpu-challenges/challenges/medium/104_min_p_sampling/challenge.html) | medium | [Solution](../catalog/104_min_p_sampling/solution.py) | 0.5941 |
| 105 | [Group Normalization](../reference/leetgpu-challenges/challenges/medium/105_group_normalization/challenge.html) | medium | [Solution](../catalog/105_group_normalization/solution.py) | 0.2290 |
| 106 | [Token Embedding Layer](../reference/leetgpu-challenges/challenges/medium/106_token_embedding_layer/challenge.html) | medium | [Solution](../catalog/106_token_embedding_layer/solution.py) | 0.5441 |
| 107 | [PPO Clipped Surrogate Loss](../reference/leetgpu-challenges/challenges/medium/107_ppo_clipped_surrogate_loss/challenge.html) | medium | [Solution](../catalog/107_ppo_clipped_surrogate_loss/solution.py) · [Shader](../catalog/107_ppo_clipped_surrogate_loss/solution.metal) | 0.1859 |
| 108 | [DPO Sequence Loss](../reference/leetgpu-challenges/challenges/medium/108_dpo_sequence_loss/challenge.html) | medium | [Solution](../catalog/108_dpo_sequence_loss/solution.py) · [Shader](../catalog/108_dpo_sequence_loss/solution.metal) | 0.1823 |
| 109 | [GRPO Surrogate Loss](../reference/leetgpu-challenges/challenges/medium/109_grpo_surrogate_loss/challenge.html) | medium | [Solution](../catalog/109_grpo_surrogate_loss/solution.py) · [Shader](../catalog/109_grpo_surrogate_loss/solution.metal) | 0.2702 |
| 110 | [Parallel Reverse Scan (GAE)](../reference/leetgpu-challenges/challenges/medium/110_gae_reverse_scan/challenge.html) | medium | [Solution](../catalog/110_gae_reverse_scan/solution.py) · [Shader](../catalog/110_gae_reverse_scan/solution.metal) | 0.4002 |
| 111 | [Softmax Attention Backward](../reference/leetgpu-challenges/challenges/medium/111_softmax_attention_backward/challenge.html) | medium | [Solution](../catalog/111_softmax_attention_backward/solution.py) | 0.3466 |
| 112 | [Attention with Sinks](../reference/leetgpu-challenges/challenges/medium/112_attention_with_sinks/challenge.html) | medium | [Solution](../catalog/112_attention_with_sinks/solution.py) | 1.7810 |
| 113 | [Layer Normalization](../reference/leetgpu-challenges/challenges/medium/113_layer_normalization/challenge.html) | medium | [Solution](../catalog/113_layer_normalization/solution.py) | 0.2746 |
| 114 | [Multi-Head Latent Attention Decode](../reference/leetgpu-challenges/challenges/hard/114_multi_head_latent_attention/challenge.html) | hard | [Solution](../catalog/114_multi_head_latent_attention/solution.py) | 0.7621 |
| 115 | [Fused QKV Projection with RoPE and KV Cache Update](../reference/leetgpu-challenges/challenges/medium/115_fused_qkv_rope_kv_cache_update/challenge.html) | medium | [Solution](../catalog/115_fused_qkv_rope_kv_cache_update/solution.py) | 0.5322 |
| 116 | [Diffusion Transformer Block](../reference/leetgpu-challenges/challenges/hard/116_dit_block/challenge.html) | hard | [Solution](../catalog/116_dit_block/solution.py) | 9.6209 |
| 117 | [Non-Maximum Suppression](../reference/leetgpu-challenges/challenges/hard/117_non_max_suppression/challenge.html) | hard | [Solution](../catalog/117_non_max_suppression/solution.py) · [Shader](../catalog/117_non_max_suppression/solution.metal) | 2.6603 |
| 118 | [Vision Transformer Patch Embedding](../reference/leetgpu-challenges/challenges/medium/118_vit_patch_embedding/challenge.html) | medium | [Solution](../catalog/118_vit_patch_embedding/solution.py) | 1.3301 |
| 119 | [Block-Sparse KV Selection Attention](../reference/leetgpu-challenges/challenges/hard/119_block_sparse_kv_selection/challenge.html) | hard | [Solution](../catalog/119_block_sparse_kv_selection/solution.py) | 2.4314 |
| 120 | [Kimi Delta Attention](../reference/leetgpu-challenges/challenges/medium/120_kimi_delta_attention/challenge.html) | medium | [Solution](../catalog/120_kimi_delta_attention/solution.py) · [Shader](../catalog/120_kimi_delta_attention/solution.metal) | 2.7500 |
| 121 | [Fused Logit Penalties](../reference/leetgpu-challenges/challenges/medium/121_fused_logit_penalties/challenge.html) | medium | [Solution](../catalog/121_fused_logit_penalties/solution.py) | 2.1257 |
| 122 | [Hamming Top-K Search](../reference/leetgpu-challenges/challenges/medium/122_hamming_top_k/challenge.html) | medium | [Solution](../catalog/122_hamming_top_k/solution.py) · [Shader](../catalog/122_hamming_top_k/solution.metal) | 0.2723 |
| 123 | [Multi-Scale Deformable Attention](../reference/leetgpu-challenges/challenges/hard/123_multi_scale_deformable_attention/challenge.html) | hard | [Solution](../catalog/123_multi_scale_deformable_attention/solution.py) · [Shader](../catalog/123_multi_scale_deformable_attention/solution.metal) | 2.4407 |
| 124 | [GELU Activation](../reference/leetgpu-challenges/challenges/easy/124_gelu/challenge.html) | easy | [Solution](../catalog/124_gelu/solution.py) | 0.1726 |
| 125 | [Scatter Add](../reference/leetgpu-challenges/challenges/medium/125_scatter_add/challenge.html) | medium | [Solution](../catalog/125_scatter_add/solution.py) | 0.3719 |

## 126–155 · Fusions, gradients, and sequence models

| ID | Problem | Level | Implementation | M3 median (ms) |
|---:|---|---|---|---:|
| 126 | [Row-wise Argmax](../reference/leetgpu-challenges/challenges/medium/126_row_argmax/challenge.html) | medium | [Solution](../catalog/126_row_argmax/solution.py) | 0.2574 |
| 127 | [Fused AdamW](../reference/leetgpu-challenges/challenges/medium/127_fused_adamw/challenge.html) | medium | [Solution](../catalog/127_fused_adamw/solution.py) · [Shader](../catalog/127_fused_adamw/solution.metal) | 0.1793 |
| 128 | [Split-KV Attention Reduction](../reference/leetgpu-challenges/challenges/medium/128_split_kv_attention_reduction/challenge.html) | medium | [Solution](../catalog/128_split_kv_attention_reduction/solution.py) | 0.4350 |
| 129 | [Fused Linear Cross Entropy](../reference/leetgpu-challenges/challenges/hard/129_fused_linear_cross_entropy/challenge.html) | hard | [Solution](../catalog/129_fused_linear_cross_entropy/solution.py) | 2.7895 |
| 130 | [Fused MoE SwiGLU Layer](../reference/leetgpu-challenges/challenges/hard/130_fused_moe_swiglu/challenge.html) | hard | [Solution](../catalog/130_fused_moe_swiglu/solution.py) | 1.6245 |
| 131 | [Fused LayerNorm Linear GELU](../reference/leetgpu-challenges/challenges/medium/131_fused_layernorm_linear_gelu/challenge.html) | medium | [Solution](../catalog/131_fused_layernorm_linear_gelu/solution.py) | 3.5130 |
| 132 | [Speculative Tree Attention](../reference/leetgpu-challenges/challenges/hard/132_speculative_tree_attention/challenge.html) | hard | [Solution](../catalog/132_speculative_tree_attention/solution.py) · [Shader](../catalog/132_speculative_tree_attention/solution.metal) | 2.5402 |
| 133 | [Multi-LoRA Batched Linear](../reference/leetgpu-challenges/challenges/medium/133_multi_lora_batched_linear/challenge.html) | medium | [Solution](../catalog/133_multi_lora_batched_linear/solution.py) | 2.5684 |
| 140 | [FP8 Block-Scaled GEMM](../reference/leetgpu-challenges/challenges/hard/140_fp8_block_scaled_gemm/challenge.html) | hard | [Solution](../catalog/140_fp8_block_scaled_gemm/solution.py) · [Shader](../catalog/140_fp8_block_scaled_gemm/solution.metal) | 1.5551 |
| 141 | [Flash Decoding Attention](../reference/leetgpu-challenges/challenges/hard/141_flash_decoding_attention/challenge.html) | hard | [Solution](../catalog/141_flash_decoding_attention/solution.py) | 0.7364 |
| 142 | [RMS Norm Backward](../reference/leetgpu-challenges/challenges/medium/142_rms_norm_backward/challenge.html) | medium | [Solution](../catalog/142_rms_norm_backward/solution.py) | 3.0535 |
| 143 | [Fused Residual Add and LayerNorm with Saved Statistics](../reference/leetgpu-challenges/challenges/hard/143_fused_residual_add_layernorm/challenge.html) | hard | [Solution](../catalog/143_fused_residual_add_layernorm/solution.py) · [Shader](../catalog/143_fused_residual_add_layernorm/solution.metal) | 0.6638 |
| 144 | [Layer Norm Backward](../reference/leetgpu-challenges/challenges/medium/144_layer_norm_backward/challenge.html) | medium | [Solution](../catalog/144_layer_norm_backward/solution.py) | 3.8465 |
| 145 | [Paged Chunked Prefill Attention](../reference/leetgpu-challenges/challenges/hard/145_paged_chunked_prefill_attention/challenge.html) | hard | [Solution](../catalog/145_paged_chunked_prefill_attention/solution.py) | 2.1637 |
| 146 | [Fused Gate-Up Projection with SiLU](../reference/leetgpu-challenges/challenges/hard/146_fused_gate_up_silu/challenge.html) | hard | [Solution](../catalog/146_fused_gate_up_silu/solution.py) · [Shader](../catalog/146_fused_gate_up_silu/solution.metal) | 0.2675 |
| 147 | [Gated Attention](../reference/leetgpu-challenges/challenges/hard/147_gated_attention/challenge.html) | hard | [Solution](../catalog/147_gated_attention/solution.py) | 1.9157 |
| 148 | [Differential Attention](../reference/leetgpu-challenges/challenges/hard/148_differential_attention/challenge.html) | hard | [Solution](../catalog/148_differential_attention/solution.py) | 3.3475 |
| 149 | [Fused RMSNorm and W8A8 Linear](../reference/leetgpu-challenges/challenges/hard/149_rmsnorm_w8a8_linear/challenge.html) | hard | [Solution](../catalog/149_rmsnorm_w8a8_linear/solution.py) | 0.9269 |
| 150 | [Distillation KL Divergence](../reference/leetgpu-challenges/challenges/medium/150_distillation_kl_divergence/challenge.html) | medium | [Solution](../catalog/150_distillation_kl_divergence/solution.py) · [Shader](../catalog/150_distillation_kl_divergence/solution.metal) | 0.3713 |
| 151 | [Muon Optimizer](../reference/leetgpu-challenges/challenges/medium/151_muon_optimizer/challenge.html) | medium | [Solution](../catalog/151_muon_optimizer/solution.py) | 5.0672 |
| 152 | [Softcapped Sliding Window Attention](../reference/leetgpu-challenges/challenges/hard/152_softcapped_sliding_window_attention/challenge.html) | hard | [Solution](../catalog/152_softcapped_sliding_window_attention/solution.py) | 31.3504 |
| 153 | [Fused RoPE Causal Attention](../reference/leetgpu-challenges/challenges/hard/153_fused_rope_causal_attention/challenge.html) | hard | [Solution](../catalog/153_fused_rope_causal_attention/solution.py) | 6.2824 |
| 154 | [Fused Mamba-2 SSD Scan](../reference/leetgpu-challenges/challenges/hard/154_fused_mamba2_ssd_scan/challenge.html) | hard | [Solution](../catalog/154_fused_mamba2_ssd_scan/solution.py) · [Shader](../catalog/154_fused_mamba2_ssd_scan/solution.metal) | 1.8851 |
| 155 | [MoE Expert Weight Gradient](../reference/leetgpu-challenges/challenges/medium/155_moe_expert_weight_gradient/challenge.html) | medium | [Solution](../catalog/155_moe_expert_weight_gradient/solution.py) | 1.1545 |
