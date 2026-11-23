from catalog.metal_shader import kernel


def solve(value, spatial_shapes, sampling_loc, attn_weight, output, num_queries,
          num_heads, head_dim, num_levels, num_points):
    kernel(__file__, "interpolate")(value, spatial_shapes, sampling_loc, attn_weight,
                                    output, num_heads, head_dim, num_levels, num_points,
                                    threads=output.numel())
