from catalog.metal_shader import kernel


def solve(teacher_logits, student_logits, kl, N, V, temperature):
    # Each group streams a vocabulary row twice and stores one scalar.
    # Recomputing scaled logits avoids materializing several full-size tensors.
    kernel(__file__, "divergence")(teacher_logits, student_logits, kl, V,
                                   1 / temperature, threads=N * 256, group_size=256)
