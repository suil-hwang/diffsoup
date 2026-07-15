# Repository Guidelines

## Project Structure & Runtime Boundaries

`python/diffsoup/` is the public Python package. It imports the compiled
`diffsoup._core` nanobind extension and provides the rasterization,
multiresolution, remeshing, point-cloud, and optimization wrappers. Native
C++17/CUDA implementation code lives in `src/`: bindings are in `src/main.cpp`,
CPU remeshing code is in `src/*.cpp`, and CUDA kernels plus launch wrappers are
under `src/cuda/`.

The actual training entry points are the scripts in `examples/`—there is no
root `train.py` in this checkout. Use `01_mip360.py`, `02_synthetic.py`, and
`03_random_init.py` for training; `04_view_results.py` through
`07_extract_bench.py` cover native viewing, FPS benchmarking, Web export, and
benchmark extraction.

Shared example logic lives in `examples/utils.py`. In particular, its `ssim`
adapter automatically uses the installed `fused_ssim` extension and falls back
to `pytorch_msssim`; training entry points should import that adapter rather
than duplicate backend logic or expose a runtime backend flag.
`submodules/fused-ssim/` is a pinned Git submodule and builds as a separate
PyTorch CUDA extension, not as part of the root scikit-build/CMake extension.

There are three viewer surfaces:

- `viewer/` is the separately built native C++ OpenGL package
  (`diffsoupviewer`), independent of CUDA at runtime.
- `py_viewer/` is the inspectable GLFW/PyOpenGL implementation. It loads
  `final_params.pt` and supports color, linear camera-Z depth, and flat
  world-space normal modes.
- `web/` is the WebGL 2/Three.js viewer and mobile benchmark.

Keep paper material in `paper/`, documentation images in `pics/`, downloaded
data in `datasets/`, training outputs in `results/`, and exported Web assets in
`web/data/`. The generated-data directories are intentionally ignored.

## Environment, Build & Smoke Commands

The documented reference environment is Ubuntu 22.04, Python 3.10, CUDA 12.4,
and an RTX 4090. The root CMake build also contains an MSVC compatibility path,
but report the OS, compiler, GPU, driver, CUDA toolkit, and PyTorch version for
every native or performance validation. On Windows use the intended activated
environment's `python`; do not assume that `python3` resolves to it.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
git submodule update --init --recursive
python -m pip install --no-build-isolation ./submodules/fused-ssim
python -m pip install -v .
python -m pip install -r requirements.txt
python examples/00_version.py
```

`python -m pip install -v .` builds and installs the CUDA/nanobind extension
through scikit-build-core. Rebuild after modifying `src/`, `CMakeLists.txt`, or
native compile definitions. The default CMake CUDA architecture is 89; set
`CMAKE_CUDA_ARCHITECTURES` explicitly when validating a different GPU target.
Rebuild `submodules/fused-ssim` after changing its pinned commit or changing
PyTorch, CUDA, the host compiler, or the target GPU architecture.

On Windows, build `fused-ssim` from an x64 Visual Studio developer prompt. If
localized `cl.exe` output triggers a `UnicodeDecodeError` in PyTorch's compiler
probe, first confirm the selected MSVC toolchain is ABI-compatible. The
validated Windows environment required `DISTUTILS_USE_SDK=1` and a temporary
`TORCH_DONT_CHECK_COMPILER_ABI=1`; do not persist the ABI-check override as a
general environment setting.

Representative workflows are:

```bash
python examples/01_mip360.py --scene_root ./datasets/360_v2/garden
python examples/02_synthetic.py --scene lego
python examples/03_random_init.py --scene lego
python -m pip install -v viewer/
python examples/04_view_results.py --ckpt results/01_mip360/garden/final_params.pt
```

For the Python viewer, install its optional runtime dependencies separately:

```bash
python -m pip install glfw PyOpenGL imgui Pillow
python -m py_viewer.cli --ckpt results/01_mip360/garden/final_params.pt --mode color
```

Use `--steps` and a separate ignored `--out_dir` for short training smoke runs;
do not overwrite an existing result directory while validating a change.

## Coding Style & Native/CUDA Conventions

Use four-space indentation. Python uses `snake_case` for functions and
variables, `PascalCase` for classes/dataclasses, and uppercase constants.
Preserve type hints and concise docstrings. Validate tensor shape, dtype,
device, contiguity, and empty-input behavior close to every native boundary.

C++/CUDA is C++17, uses descriptive lower-snake-case functions and namespaces
such as `diffsoup::cuda`, and should match surrounding formatting. Kernel
launches must use the caller/current PyTorch CUDA stream; avoid implicit
default-stream work, device-wide synchronization, per-call allocation, and
redundant zeroing. Workspace, accumulation-plan, and fragment reuse must remain
safe across shapes, devices, forward/backward calls, and non-default streams.
Do not trade exact forward behavior or validated gradient tolerances for a
timing improvement.

Keep the shared SSIM contract unchanged: arguments are prediction first and
target second, tensors are NCHW with values in `[0, 1]`, and fused execution
uses `padding="valid"`. The fused implementation differentiates only its first
argument, so reversing the inputs silently removes the prediction gradient.
Keep `pytorch_msssim` as the automatic fallback with `data_range=1.0`; use
`prediction.requires_grad` together with `torch.is_grad_enabled()` when
deciding whether fused backward data is needed.

GLSL files in `py_viewer/` target OpenGL 4.1 core. Preserve the geometry-pass
MRT contract, alpha discard rule, and default color output when adding a debug
mode. Checkpoint loading should keep `torch.load(..., weights_only=True)` unless
a narrowly documented compatibility reason requires otherwise.

No formatter or linter is enforced. Keep imports grouped, avoid unrelated
mechanical rewrites, and run `git diff --check` before handoff.

## Testing, Profiling & Validation

There is currently no committed automated test suite or coverage threshold.
This checkout can contain local CUDA tests under `tests/`, but `tests/` is
ignored by `.gitignore` and can be invisible to ordinary `rg --files`. Use
`Get-ChildItem tests` on PowerShell or `rg --no-ignore --files tests` before
claiming tests are absent. If a new test is meant to be committed, update the
ignore rules deliberately and confirm it appears in `git status`.

The current CUDA optimization regression file can be run with:

```bash
python -m pytest -q -p no:cacheprovider tests/test_cuda_optimizations.py
python -m compileall -q python py_viewer examples
pyrefly check examples/utils.py examples/01_mip360.py examples/02_synthetic.py examples/03_random_init.py
python examples/00_version.py
```

CUDA tests require a rebuilt extension and a real CUDA device. Kernel changes
must cover representative levels, feature dimensions, empty geometry,
forward/backward parity, reused auxiliary fragments/plans, and non-default
streams. Synchronize only at measurement or assertion boundaries.

SSIM changes require forward and prediction-gradient parity against
`pytorch_msssim` with valid padding. Benchmark forward plus backward after
warm-up with CUDA events and report peak allocated memory for garden-resolution
B1, B2, and B4 inputs. Because Windows WDDM slowdown can appear only after the
step-5,000 multiresolution lift, run garden B4 through at least step 5,400 in a
separate ignored output directory before claiming long-run stability.

For performance work, establish a correctness baseline first, warm up kernels,
use CUDA events or explicit synchronization around the measured interval, and
report tensor shapes plus hardware/software details. Use Nsight Compute on a
small targeted workload when investigating occupancy, memory traffic, launch
overhead, or redundant initialization. Always pair speed numbers with exact or
tolerance-based output and gradient comparisons.

Rendering changes require a real OpenGL context and a trusted representative
checkpoint, not only shader text inspection. Capture before/after images or
hashes and frame timings. In `py_viewer`, depth means positive camera-axis
linear Z; `auto` changes only display contrast, while `clip` preserves a fixed
near/far mapping. Normal output is flat world-space `RGB = XYZ`:
`face-forward` is view-dependent, whereas `oriented` preserves triangle
winding. Do not describe either as smoothed, camera-space, or lossless
floating-point output.

## Commit & Pull Request Guidelines

Use short imperative subjects such as `Add ...`, `Fix ...`, `Optimize ...`, or
`Refine ...`, and keep each commit focused. Preserve unrelated user changes in
a dirty worktree. Pull requests should explain motivation and affected paths,
list exact validation commands and hardware/CUDA/OpenGL details, and include
metric deltas or screenshots for CUDA, viewer, Web, or rendering changes. Do
not commit datasets, checkpoints, result images, benchmark dumps, build
products, virtual environments, or generated Web data.
When updating `fused-ssim`, commit the intended gitlink change together with
any necessary `.gitmodules` change, verify `git submodule status`, and keep
submodule `build/`, wheel, and egg-info artifacts untracked.
