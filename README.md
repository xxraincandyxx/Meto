# Metal Lab

Learn Apple GPU compute in VS Code on your Mac. Start by adding two vectors, check the answer against a CPU calculation, then measure what changes when one GPU thread handles four numbers. The course follows all **147 challenges** in the pinned LeetGPU catalog, in their original numeric order. [Browse the curriculum](docs/curriculum.md)

## Your first run

Open this folder in VS Code. Press **⌘⇧B**, enter **01**, and press Return. The default task builds the Swift host, compiles the shader locally, dispatches it to Metal, and checks five input lengths. In the integrated terminal run

```sh
./metal-lab check 01
```

Open the baseline and find the line that writes the sum. The input of length 257 makes the last threadgroup only partly useful. In a practice copy, change the sum to a difference and run the check. The first mismatched index shows how the runner catches an incorrect kernel. Restore the sum and compare both versions with `./metal-lab bench 01`. [Baseline](lessons/01-vector-add/baseline.metal) · [Optimized kernel](lessons/01-vector-add/solution.metal) · [First six labs](docs/first-six.md)

## Set up a Mac

Use an Apple silicon Mac with a working Metal device, VS Code, and Apple's Command Line Tools. Install the tools with `xcode-select --install` if `xcrun --find swiftc` fails. The Swift host compiles shader source through Metal at launch. This works with the standalone Command Line Tools. [Apple runtime compiler documentation](https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:))

| What you want to run | Setup |
|---|---|
| First six native lessons | Open this folder and run `./metal-lab doctor`. |
| Full catalog | Have `uv` or Python 3.13 available, then run `./metal-lab setup`. With Homebrew, `brew install uv` supplies the installer. |
| Another checkout | Use `git clone --recurse-submodules`, or run `git submodule update --init` inside an existing clone. |

Setup creates a project-local Python environment under `.build/catalog-env`. With `uv` it can also fetch Python 3.13. For an existing interpreter use `PYTHON=/path/to/python3.13 ./metal-lab setup` on a Mac without `uv`. After setup the exercises and verification run offline.

## The edit–check–measure loop

| VS Code task | Integrated terminal | Result |
|---|---|---|
| Metal Lab: List lessons | `./metal-lab list` | See all IDs and titles. |
| Metal Lab: Read problem | `./metal-lab show 04` | Read the prompt and locate the examples and implementation. |
| Metal Lab: Create practice copy | `./metal-lab practice 04` | Create editable files next to the lesson. Repeating the command preserves your edits. |
| Metal Lab: Check practice | `./metal-lab check 04 practice` | Build your version and compare its outputs with the CPU. |
| Metal Lab: Check solution | `./metal-lab check 04` | Check the supplied solution on example and edge cases. |
| Metal Lab: Benchmark solution | `./metal-lab bench 04` | Validate the workload, warm up, and print a median. |
| Metal Lab: Check all lessons | `./metal-lab check all` | Run the full local correctness suite. |

Find the tasks under **Terminal → Run Task**. Practice copies start from the simple baseline in lessons 1–6 and the worked solution in later lessons. Files named `practice.metal` and `practice.py` are yours to edit and commit. To compare a change with the supplied version, run `bench ID practice` and `bench ID` on the same input.

## How to read the solutions

**Lessons 1–6** expose the full Metal path. Each Swift lesson defines the shapes, CPU calculation, buffer order, and threadgroup sizes for its two shaders. The shared runtime builds compute pipelines and checks completion before Swift reads a buffer. [Metal runtime](Sources/MetalRuntime.swift)

**Lessons 7–155** keep the upstream IDs, including eight gaps. Each lesson has an independently editable `solution.py` that receives tensors already on the GPU. Where a solution dispatches custom Metal, open the adjacent `.metal` file to study its thread mapping, memory accesses, and synchronization. Other lessons compose PyTorch's Metal-backed operations for work such as GEMM, FFT, or attention. The curriculum links to the actual implementation type for each ID. PyTorch supplies the MPS backend and the interface for launching custom shaders. [PyTorch MPS](https://docs.pytorch.org/docs/2.15/mps.html) · [Custom Metal shaders](https://docs.pytorch.org/docs/2.15/generated/torch.mps.compile_shader.html)

The extended runner generates inputs from the pinned reference, evaluates the reference on the CPU, and runs the local solution on independent GPU buffers. It checks all declared outputs, including mutated inputs, using exact integer comparisons and each challenge's floating-point tolerances. `smoke all` runs the short native checks and the example from each extended lesson. [Catalog runner](Tools/catalog.py)

## Measuring an optimization

The native benchmark compares a baseline and an optimized kernel at the same shape. It reports median **GPU command-buffer time** over 15 warmed runs. The extended benchmark reports **synchronized elapsed time**, including Python dispatch and any allocations inside the solution. By default it picks the largest example or functional fixture containing at most 32 MiB of tensor arguments, or the smallest available fixture when all are larger. Each run prints the fixture and tensor shapes. Use `./metal-lab bench all` to time the whole course. For a larger upstream workload, run `./metal-lab bench 17 --performance`. Check the problem's declared performance shape first. Some transformer inputs alone occupy gigabytes. CPU verification, transfers into the harness, and shader compilation happen before the timed samples.

Matrix sizes, strides, chip generation, thermal state, and the cost of a dispatch can all change which implementation wins. Use the recorded measurements to select an experiment, then measure it on your own workload. [Native benchmark notes](docs/first-six.md) · [Lessons 7–50](docs/benchmarks-07-50.md) · [Lessons 51–101](docs/benchmarks-51-101.md) · [Lessons 102–155](docs/benchmarks-102-155.md)

## Reference

AlphaGPU's problem statements, examples, and CPU references live in an unmodified Git submodule pinned at `3b2ec31`. Their license is included in the checkout. The local runners, explanations, and solutions live in this repository. [Upstream repository](https://github.com/AlphaGPU/leetgpu-challenges) · [Reference license](reference/leetgpu-challenges/LICENSE)
