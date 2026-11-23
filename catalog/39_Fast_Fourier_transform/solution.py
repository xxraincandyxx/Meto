"""Transform interleaved real and imaginary components into Fourier coefficients."""

import torch


def solve(signal, spectrum, N):
    # Each adjacent pair of float32 elements is already one complex64 value.
    # Reinterpret their storage on both sides of the GPU FFT. This saves packing
    # the input and splitting the output into two additional full-size tensors.
    samples = torch.view_as_complex(signal.view(N, 2))
    frequencies = torch.view_as_complex(spectrum.view(N, 2))
    torch.fft.fft(samples, out=frequencies)
