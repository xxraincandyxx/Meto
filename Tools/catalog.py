"""Run the pinned CPU oracle and an independently editable Metal solution locally."""
import argparse
import copy
import importlib.util
import os
import statistics
import sys
import time

from course import ROOT, entries
# Let lessons import common GPU helpers even when launched from VS Code.
sys.path.insert(0, str(ROOT))


def load(path):
    spec = importlib.util.spec_from_file_location(f"lab_{path.parent.name}_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(case, device):
    result = {}
    for key, value in case.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().to(device, copy=True).contiguous()
        elif isinstance(value, torch.nn.Module):
            result[key] = copy.deepcopy(value).to(device).eval()
        else:
            result[key] = value
    return result


def validate(solve, challenge, case):
    cpu = prepare(case, "cpu")
    gpu = prepare(case, "mps")
    signature = challenge.get_solve_signature()
    outputs = [key for key, (_, mode) in signature.items() if mode in ("out", "inout")]
    if not outputs:
        raise RuntimeError("Missing output declaration in reference signature.")
    for key in outputs:
        # Unused expert capacity retains its initial
        # values. Other pure outputs get a sentinel to expose missed writes.
        partial = (challenge.name, key) == ("MoE Token Dispatch", "dispatched_x")
        if signature[key][1] == "out" and not partial:
            gpu[key].fill_(float("nan") if gpu[key].is_floating_point() else 11)
    challenge.reference_impl(**cpu)
    solve(**gpu)
    torch.mps.synchronize()
    for key in outputs:
        actual, expected = gpu[key].cpu(), cpu[key]
        torch.testing.assert_close(actual, expected, atol=challenge.atol, rtol=challenge.rtol,
                                   msg=lambda detail: f"{key}: {detail}")
    return gpu


def exercise(path, args):
    folder = ROOT / "catalog" / path.parent.name
    source = folder / f"{args.variant}.py"
    torch.manual_seed(2026)
    challenge = load(path).Challenge()
    solve = load(source).solve
    cases = [challenge.generate_example_test()]
    if args.action == "check":
        cases.extend(challenge.generate_functional_test())
    if args.action == "bench" and args.performance:
        cases = [challenge.generate_performance_test()]
    for i, case in enumerate(cases):
        try:
            validate(solve, challenge, case)
        except Exception as exc:
            raise RuntimeError(f"case {i} ({'example' if i == 0 else 'functional'}): {exc}") from exc
    if args.action == "bench":
        case = cases[0]
        gpu = prepare(case, "mps")
        # Restore mutated buffers before each run. Otherwise e.g. an in-place sort
        # would measure an already-sorted input after the first iteration.
        original = {k: gpu[k].clone() for k, (_, mode) in challenge.get_solve_signature().items()
                    if mode == "inout"}
        times = []
        shapes = ", ".join(f"{k}={tuple(v.shape)}" for k, v in gpu.items() if isinstance(v, torch.Tensor))
        print("  " + shapes)
        for i in range(17):
            for key, value in original.items():
                gpu[key].copy_(value)
            torch.mps.synchronize()
            start = time.perf_counter_ns()
            solve(**gpu)
            torch.mps.synchronize()
            if i >= 2:
                times.append((time.perf_counter_ns() - start) / 1e6)
        print(f"  {statistics.median(times):.4f} ms median synchronized latency (15 runs)")
    print(f"  PASS {len(cases)} CPU/GPU comparison(s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "smoke", "bench"])
    parser.add_argument("id", nargs="?", default="all")
    parser.add_argument("variant", nargs="?", default="solution", choices=["solution", "practice"])
    parser.add_argument("--performance", action="store_true", help="use upstream performance workload")
    args = parser.parse_args()
    lessons = entries()
    global torch
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("Metal is unavailable to PyTorch. Run locally on a supported Mac with GPU access.")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK to run checks entirely on Metal.")
    if args.id == "all" and args.action in ("check", "smoke"):
        selected = [p for i, p in lessons.items() if i > 6]
    elif args.id.isdigit() and int(args.id) in lessons:
        selected = [lessons[int(args.id)]]
    else:
        raise ValueError(f"Unknown lesson {args.id!r}. See ./metal-lab list")
    failures = 0
    with torch.inference_mode():
        for path in selected:
            print(path.parent.name, flush=True)
            try:
                exercise(path, args)
            except Exception as exc:
                failures += 1
                print(f"  FAIL {exc}", file=sys.stderr, flush=True)
            torch.mps.empty_cache()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        sys.exit(str(exc))
