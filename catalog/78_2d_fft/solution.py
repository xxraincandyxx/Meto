import torch


def solve(signal, spectrum, M, N):
    # The API supplies interleaved real/imaginary float pairs. Reinterpret the
    # storage as complex numbers, so neither packing nor unpacking copies data.
    image = torch.view_as_complex(signal.view(M, N, 2))
    target = torch.view_as_complex(spectrum.view(M, N, 2))
    torch.fft.fft2(image, out=target)
