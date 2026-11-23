"""Invert each RGB channel in an interleaved RGBA image."""

from catalog.metal_shader import kernel


def solve(image, width, height):
    kernel(__file__, "invert")(image, threads=width * height)
