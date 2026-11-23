import math
from catalog.metal_shader import kernel


def solve(prompts, output, weights, batch_size):
    # The attention amplitude is specified by a decimal logit gap at peak
    # phase 0.3; the normalized two-dimensional heads supply a factor of 2.
    omega = 2 * math.pi / 19
    scale = 0.5 * math.log(10) / (math.cos(0.3 * omega) - math.cos(0.7 * omega))
    kernel(__file__)(prompts, weights, output, scale, threads=batch_size)
