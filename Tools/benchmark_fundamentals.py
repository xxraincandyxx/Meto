"""Reproduce the bounded working sets in docs/benchmarks-07-50.md on a local Mac."""

from pathlib import Path
import argparse
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tools import catalog as lab


def workload(number):
    """Create an original, deterministic CPU workload in the public solve argument order."""
    n = 262144
    f = torch.randn
    out = torch.empty
    i32 = torch.int32

    def integers(*shape, high=64):
        return torch.randint(0, high, shape, dtype=i32)

    def empty_int(*shape):
        return torch.empty(shape, dtype=i32)

    if number == 7:
        return "512×512 RGBA", [torch.randint(0, 256, (n * 4,), dtype=torch.uint8), 512, 512]
    if number in (8, 31):
        args = [f(512, 512), out(512, 512), 512]
        if number == 8:
            args.insert(1, f(512, 512))
        return "512×512", args
    if number == 9:
        return "65536, filter 257", [f(65536), f(257), out(65280), 65536, 257]
    if number in (10, 28):
        side = 256 if number == 28 else 250
        return "256×256, filter 7×7", [f(256**2), f(49), out(side**2), 256, 256, 7, 7]
    if number == 11:
        return "32³, filter 3³", [f(32, 32, 32), f(3, 3, 3), out(30, 30, 30), 32, 32, 32, 3, 3, 3]
    if number == 12:
        return "256 tokens, 8 heads × 32", [f(256, 256), f(256, 256), f(256, 256), out(256, 256), 256, 256, 8]
    if number == 13:
        return "262144, 256 bins", [integers(n, high=256), empty_int(256), n, 256]
    if number == 14:
        return "1024 agents", [f(4096), out(4096), 1024]
    if number in (15, 19):
        return "262144", [f(n), n]
    if number in (16, 21, 23):
        return "262144", [f(n), out(n), n]
    if number in (17, 27):
        return "1048576", [f(4 * n), f(4 * n), out(1), 4 * n]
    if number == 18:
        matrix = f(256, 512)
        matrix[torch.rand_like(matrix) < 0.65] = 0
        return "256×512, ~65% zero", [matrix, f(512), out(256), 256, 512, matrix.count_nonzero().item()]
    if number == 20:
        return "2048 points, k=5, 8 steps", [
            f(2048),
            f(2048),
            empty_int(2048),
            f(5),
            f(5),
            out(5),
            out(5),
            2048,
            5,
            8,
        ]
    if number == 22:
        return "256³, FP16", [f(256, 256).half() for _ in range(3)] + [256, 256, 256, 0.5, 0.25]
    if number == 24:
        return "262144, 20 rounds", [integers(n, high=2**30), out(n, dtype=torch.uint32), n, 20]
    if number == 25:
        return "2048×512", [f(2048, 512), integers(2048, high=512), out(1), 2048, 512]
    if number == 26:
        return "128 queries, 256 keys, 4 heads × 32", [
            f(128, 4, 32),
            f(256, 4, 32),
            f(256, 4, 32),
            out(128, 4, 32),
            128,
            256,
            4,
            32,
        ]
    if number == 29:
        return "262144, top 32", [f(n), out(32), n, 32]
    if number == 30:
        return "16 batches, 64³", [f(16, 64, 64), f(16, 64, 64), out(16, 64, 64), 16, 64, 64, 64]
    if number == 32:
        left, right = [torch.randint(-8, 9, (256, 256), dtype=torch.int8) for _ in range(2)]
        return "256³, INT8", [left, right, torch.empty_like(left), 256, 256, 256, 0.1, 0.2, 0.25, 0, 1, 0]
    if number == 33:
        return "1024 samples, 16 features", [f(1024, 16), f(1024), out(16), 1024, 16]
    if number == 34:
        return "16 samples, 8 features", [
            torch.eye(8).repeat(2, 1),
            torch.tensor([0.0, 1.0] * 8),
            out(8),
            16,
            8,
        ]
    if number == 35:
        return "262144", [f(n), out(1), -2.0, 3.0, n]
    if number == 36:
        keys = torch.randint(0, 2**32, (n,), dtype=torch.int64).to(torch.uint32)
        return "262144 unsigned keys", [keys, torch.empty_like(keys), n]
    if number == 37:
        return "128×128, power 7", [f(128**2) / 16, out(128**2), 128, 7]
    if number == 38:
        return "1024 XYZ points", [f(3072), empty_int(1024), 1024]
    if number == 39:
        return "65536 complex samples", [f(131072), out(131072), 65536]
    if number == 40:
        return "1024×128", [f(1024, 128), f(128), f(128), out(1024, 128), 1024, 128, 1e-5]
    if number == 41:
        return "128 samples, 256→128", [f(128, 256), torch.nn.Linear(256, 128), out(128, 128)]
    if number == 42:
        return "4×8×64×64, 3×3 stride 2", [f(4 * 8 * 64**2), out(4 * 8 * 32**2), 4, 8, 64, 64, 3, 2, 1]
    if number in (43, 44, 45):
        shape = {43: (n,), 44: (512, 512), 45: (64, 64, 64)}[number]
        return "×".join(map(str, shape)), [integers(*shape), empty_int(1), *shape, 17]
    if number == 46:
        return "64×64, open grid", [torch.zeros(4096, dtype=i32), empty_int(1), 64, 64, 0, 0, 63, 63]
    if number == 47:
        return "262144, middle half", [integers(n), empty_int(1), n, n // 4, 3 * n // 4 - 1]
    if number == 48:
        return "512², middle 256²", [integers(512, 512), empty_int(1), 512, 512, 128, 383, 128, 383]
    if number == 49:
        return "64³, middle 32³", [integers(64, 64, 64), empty_int(1), 64, 64, 64, 16, 47, 16, 47, 16, 47]
    if number == 50:
        return "65536", [f(65536), 1.2, -0.5, out(65536), 65536, 1e-5]
    raise ValueError(number)


def timed(solve, gpu, original):
    samples = []
    for i in range(17):
        for name, tensor in original.items():
            gpu[name].copy_(tensor)
        torch.mps.synchronize()
        start = time.perf_counter_ns()
        solve(**gpu)
        torch.mps.synchronize()
        if i >= 2:
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


def comparison(number, module):
    # These alternatives exercise the same inputs, output contracts and device.
    if number == 15:
        return lab.load(Path(module.__file__).with_name("radix.py")).solve
    if number == 25:
        return lambda logits, true_labels, loss, N, C: loss.copy_(
            torch.nn.functional.cross_entropy(logits, true_labels.long())
        )
    if number == 27:
        return lambda predictions, targets, mse, N: mse.copy_(
            torch.nn.functional.mse_loss(predictions, targets)
        )
    if number == 34:
        return module.general_newton
    if number == 39:

        def packed_fft(signal, spectrum, N):
            pairs = signal.view(N, 2)
            bins = torch.fft.fft(torch.complex(pairs[:, 0], pairs[:, 1]))
            spectrum.copy_(torch.stack((bins.real, bins.imag), dim=1).flatten())

        return packed_fft
    if number == 41:
        return lambda input, model, output: output.copy_(model(input))
    if number == 46:

        def global_grid(grid, result, rows, cols, start_row, start_col, end_row, end_col):
            module.device_frontiers(
                grid, result, rows, cols, start_row * cols + start_col, end_row * cols + end_col
            )

        return global_grid
    return None


@torch.inference_mode()
def main():
    lab.torch = torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", type=int, help="run individual lesson IDs")
    parser.add_argument(
        "--compare", action="store_true", help="time the alternative kernels from the tuning report"
    )
    args = parser.parse_args()
    selected = set(args.ids) or set(range(7, 51))
    compare = args.compare
    print("| ID | Workload | Median ms |\n|---:|---|---:|", flush=True)
    for number, path in lab.entries().items():
        if number not in selected:
            continue
        torch.manual_seed(2026 + number)
        description, arguments = workload(number)
        challenge = lab.load(path).Challenge()
        signature = challenge.get_solve_signature()
        case = dict(zip(signature, arguments, strict=True))
        module = lab.load(lab.ROOT / "catalog" / path.parent.name / "solution.py")
        solve = comparison(number, module) if compare else module.solve
        if solve is None:
            continue
        lab.validate(solve, challenge, case)
        gpu = lab.prepare(case, "mps")
        original = {k: gpu[k].clone() for k, (_, mode) in signature.items() if mode == "inout"}
        latency = timed(solve, gpu, original)
        print(f"| {number} | {description} | {latency:.3f} |", flush=True)


if __name__ == "__main__":
    main()
