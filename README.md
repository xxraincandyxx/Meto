<div align="center">

# Meto

**Learn Apple GPU compute on your own Mac.**

All **147 LeetGPU challenges** in upstream order — each lesson ships a plain
baseline, a worked solution, and an editable practice copy, graded against the
challenge's own CPU reference and timed repeatably on your machine.
Fully offline: no cloud, no account, no network calls.

[Quick start](#quick-start) · [Workflow](#everyday-workflow) · [Two tracks](#two-tracks) · [Docs](#documentation) · [License](#upstream-problems-and-license)

**English** · [繁體中文](README.zh-TW.md)

</div>

---

> IDs 97, 100, and 134–139 are upstream numbering gaps. 61 of the catalog
> lessons ship a hand-written shader alongside their Python solution.

## Quick start

```sh
git clone --recurse-submodules <repository-url>
cd meto
./meto setup     # fetch the problem submodule, build a local Python env, self-check
```

Already inside the folder? Run `git submodule update --init`, then `./meto setup`.

**Run your first lesson** — open the folder in VS Code and press **⌘⇧B**, typing `01`. Or from a terminal:

```sh
./meto check 01
```

That single command builds the Swift host, compiles the shader at launch, dispatches the kernel on the GPU, and compares five input lengths against a CPU calculation. Length 257 leaves the last threadgroup partly unused, so it is the case that catches a missing tail.

**Learn by editing.** `./meto practice 01` creates `lessons/01-vector-add/practice.metal`; turn the sum into a difference, then run `./meto check 01 practice` — the first mismatched index is the bug the runner found for you. Restore the sum and compare against the shipped version with `./meto bench 01`. `practice.metal` and `practice.py` are yours to commit, and repeating `practice` never overwrites your edits.

## Requirements

| What you need | Why | How to get it |
|---|---|---|
| Apple silicon Mac with a Metal device | Every check runs on the local GPU | `./meto doctor` prints the device name |
| Apple Command Line Tools | Swift host plus runtime shader compilation | `xcode-select --install` (when `xcrun --find swiftc` fails) |
| VS Code | Wraps the CLI as Run Tasks | Open this folder, Terminal → Run Task |
| `uv` or Python 3.13 | PyTorch MPS backend for lessons 07–155 | `brew install uv`, or `PYTHON=/path/to/python3.13 ./meto setup` |

`setup` installs dependencies into the project-local `.build/catalog-env` (PyTorch 2.15 + NumPy), and VS Code already points at that interpreter. After setup, exercises, grading, and timing all work offline.

## Everyday workflow

| Goal | Command | What you get |
|---|---|---|
| Browse the course | `./meto list` | Every ID and title |
| Read a problem | `./meto show 04` | Statement plus where the examples and code live |
| Start editing | `./meto practice 04` | `practice.metal` / `practice.py` next to the lesson |
| Check your version | `./meto check 04 practice` | First mismatched index, with expected and actual values |
| Check the shipped solution | `./meto check 04` | The example plus every functional case |
| Time it | `./meto bench 04` | Validates, warms up, then prints a median |
| Quick pass over the course | `./meto smoke all` | Native cases + one example per lesson |
| Full correctness suite | `./meto check all` | Native track + catalog track |
| Time the whole course | `./meto bench all` | One line per lesson |
| Upstream workload sizes | `./meto bench 17 --performance` | The declared performance shape (some transformer inputs reach GB — read the problem first) |

## Two tracks

**01–06 · native Metal.** Each lesson owns its shapes, CPU oracle, and launch geometry; a shared Swift runtime compiles the shader at launch, binds buffers, dispatches threadgroups, waits for completion, and compares the output. Trace one dispatch from Swift to GPU before moving on. [Metal runtime](Sources/MetalRuntime.swift) · [First six labs](docs/first-six.md)

**07–155 · catalog.** Each lesson is an editable `solution.py` that receives tensors already on the GPU. 61 lessons dispatch hand-written Metal through `torch.mps.compile_shader`; the rest compose PyTorch MPS ops for GEMM, FFT, and attention. Open the sibling `.metal` file to study thread mapping, memory access, and synchronization. [Catalog runner](Tools/catalog.py) · [Shader loader](catalog/metal_shader.py)

## Grading and timing

- Inputs come from the pinned upstream fixtures; the reference runs on CPU, your solution runs on independent GPU buffers.
- Output buffers are filled with poison values first (`NaN` for floats, `11` for integers), so a missing write cannot hide behind a zero-filled allocation.
- Every declared output is compared, including inputs mutated in place. Integers must match exactly; floats use each challenge's own `atol` / `rtol`.
- The catalog track exits if `PYTORCH_ENABLE_MPS_FALLBACK=1`, so nothing can quietly fall back to CPU and still pass.
- Native lessons report **median GPU command-buffer time** at one shape after warm-up (15 samples). Catalog lessons report **median synchronized wall time**, including Python dispatch and allocations inside `solve`.
- The default fixture is the largest case whose tensor arguments fit in 32 MiB, or the smallest when all are larger; each run prints the fixture and shapes.
- CPU verification, uploads, and shader compilation happen outside the timed window, and in-place inputs are restored between runs.
- Chip generation, thermal state, matrix sizes, and dispatch cost all change the winner. Use the documented numbers to pick an experiment, then re-measure on your own workload.

## Repository layout

```text
Meto/
├── meto            Single entry point for every command
├── Makefile         Build rules for the Swift host
├── Sources/         GPU device, buffers, pipeline cache, dispatch, comparison
├── lessons/         01–06: baseline.metal / solution.metal + Swift dispatch
├── catalog/         07–155: solution.py, plus solution.metal for 61 lessons
├── Tools/           Course index and the CPU/GPU grading runner
├── docs/            Curriculum, first six labs, three benchmark reports
└── reference/       Upstream problems (pinned Git submodule)
```

## Documentation

| Contents | Where |
|---|---|
| All 147 IDs, levels, implementation links, measured M3 times | [docs/curriculum.md](docs/curriculum.md) |
| One dispatch traced from Swift to GPU, with questions per lesson | [docs/first-six.md](docs/first-six.md) |
| Measured numbers and tuning notes | [07–50](docs/benchmarks-07-50.md) · [51–101](docs/benchmarks-51-101.md) · [102–155](docs/benchmarks-102-155.md) |
| Every command and flag | `./meto help` |

## Upstream problems and license

Problem statements, examples, and CPU references come from an unmodified Git submodule at `reference/leetgpu-challenges`, pinned to commit `3b2ec31`; its license arrives with the checkout. The runners, explanations, and solutions live in this repository. [AlphaGPU/leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) · [Upstream license](reference/leetgpu-challenges/LICENSE)
