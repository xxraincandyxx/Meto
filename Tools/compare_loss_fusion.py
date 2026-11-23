"""Measure four fused loss kernels against the pinned eager MPS formulas."""
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Tools import catalog as lab


def compare(number):
    path = lab.entries()[number]
    problem = lab.load(path).Challenge()
    if number == 150:
        # The full upstream case alone has two 1 GiB inputs. Keep the unfused
        # formula's additional temporaries comfortable on an entry-level Mac.
        rows, vocab = 256, 32000
        case = dict(teacher_logits=torch.randn(rows, vocab), student_logits=torch.randn(rows, vocab),
                    kl=torch.empty(rows), N=rows, V=vocab, temperature=2.0)
    else:
        case = problem.generate_performance_test()
    args = {key: value.to("mps") if isinstance(value, torch.Tensor) else value
            for key, value in case.items()}
    output = next(key for key, (_, mode) in problem.get_solve_signature().items() if mode == "out")
    optimized = lab.load(lab.ROOT / "catalog" / path.parent.name / "solution.py").solve
    functions = (problem.reference_impl, optimized)
    problem.reference_impl(**case)
    times = ([], [])
    for method in functions:
        method(**args)
        torch.testing.assert_close(args[output].cpu(), case[output], atol=problem.atol, rtol=problem.rtol)
    for iteration in range(18):
        # Alternate order to spread clock and thermal drift across implementations.
        for idx in (0, 1) if iteration % 2 == 0 else (1, 0):
            torch.mps.synchronize()
            start = time.perf_counter_ns()
            functions[idx](**args)
            torch.mps.synchronize()
            if iteration >= 3:
                times[idx].append((time.perf_counter_ns() - start) / 1e6)
    eager, fused = map(statistics.median, times)
    print(f"{number}: eager {eager:.4f} ms, fused {fused:.4f} ms, speedup {eager / fused:.2f}x")


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise SystemExit("Run on a Mac with an available Metal device.")
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise SystemExit("Unset PYTORCH_ENABLE_MPS_FALLBACK to time GPU operations.")
    torch.manual_seed(2026)
    with torch.inference_mode():
        for number in (107, 108, 109, 150):
            compare(number)
