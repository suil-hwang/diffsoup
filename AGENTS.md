# Repository Guidelines

## Current Checkout, Structure, and Runtime Boundaries

Treat the checked-out source tree as authoritative. There is no root
`train.py`, and this checkout does not contain `examples/02_synthetic.py` even
though some README text and example comments still mention it. The tracked
training entry points are:

- `examples/01_mip360.py`: the baseline COLMAP/MipNeRF-360 trainer.
- `examples/02_mip360_test.py`: the current MipNeRF-360 experimental trainer.
  It adds a separately controlled LR schedule horizon, a nonempty-output guard,
  a configurable target face count, detailed loss telemetry, and fixed-weight
  ARAG depth/normal supervision. Despite the `_test` suffix, this is a full
  trainer.
- `examples/02_mip360_test_profile.py`: the matching profiling trainer. It keeps
  training behavior aligned with `02_mip360_test.py`, always disables tqdm, and
  writes per-step CUDA/wall/memory telemetry to `train_profile.jsonl`.
- `examples/03_random_init.py`: random point/triangle initialization for
  NeRF-Synthetic-style scenes; there is currently no tracked mesh-initialized
  synthetic trainer.

`examples/04_view_results.py` through `07_extract_bench.py` cover native
viewing, native FPS benchmarking, Web export, and mobile-benchmark extraction.
`examples/08_prepare_geometry_priors.py` prepares a complete scene-level ARAG
depth/normal prior layout for `02_mip360_test.py`. Shared COLMAP/camera/data,
SSIM, visibility, pruning, and screen-space splitting helpers live in
`examples/utils.py`. Its Mip-NeRF360 loader preserves per-image COLMAP
intrinsics as `frame["K"]` and stacked `Ks`; `K` remains the first selected
intrinsic only for legacy checkpoint/viewer compatibility.

`python/diffsoup/` is the public Python package. It imports the compiled
`diffsoup._core` nanobind extension and exposes:

- `rasterize.py`: fragment generation, stochastic opacity/depth testing,
  opacity and edge-gradient surrogates, SH2 view encoding, and triangle counts;
- `multires.py`: triangular feature lattices, accumulation, interpolation,
  reusable raster fragments, and `ColorMLP`;
- `remesh.py` and `point3d.py`: CPU remeshing wrappers and point-cloud soup
  initialization/cleanup;
- `surface.py`, `regularization.py`, and `priors.py`: sparse differentiable
  expected surfaces, depth/normal losses, and aligned prior loading/sampling;
- `optimize.py`: the isotropic-last-axis `VectorAdam` optimizer.

Native implementation code lives in `src/`. `src/main.cpp` binds PyTorch
tensors as nanobind/DLPack array views; it is not a libtorch/ATen extension.
CPU remeshing is in `src/remesh*.cpp`, and CUDA fragment, multiresolution, and
SH2 kernels are in `src/cuda/`. Python wrappers own tensor validation and pass
the caller's current PyTorch CUDA stream as a raw handle; `_core` is an internal
boundary and should not be called directly by application code.

The three pinned submodules are separate from the root CMake target:

- `submodules/fused-ssim/` builds as its own PyTorch CUDA extension and is used
  automatically by the examples when importable.
- `submodules/arag/` supplies offline geometry-prior models and code for
  `08_prepare_geometry_priors.py`; its model checkpoint is a separate,
  intentionally untracked download. Its local `.gitignore` excludes Python
  bytecode/tool caches and downloaded `.pth` weights; to share that rule with
  future clones, commit it inside the submodule and then update the gitlink.
- `submodules/nvdiffrast/` is pinned source for separate experimentation. No
  tracked root package or example currently imports or links it, so do not add
  it to the root build implicitly.

There are three viewer surfaces:

- `viewer/` is the separately built `diffsoupviewer` C++/OpenGL 4.1 package.
  It accepts NumPy mesh/LUT/MLP arrays rather than loading checkpoints itself,
  has no CUDA runtime dependency, and provides interactive color rendering and
  a visible-window FPS benchmark.
- `py_viewer/` is the inspectable GLFW/ModernGL 4.1 implementation. It loads a
  checkpoint with `weights_only=True` and supports color, positive linear
  camera-Z depth, flat world-space normals, screenshots, and a hidden-context
  benchmark API. Its current pyimgui GLFW integration still uses PyOpenGL
  internally, but the DiffSoup render/resource path is ModernGL-managed.
- `web/` is the WebGL 2/Three.js viewer and mobile benchmark. It consumes
  assets exported by `06_export_web.py`; the desktop viewer loads
  `web/data/models.json`, while `benchmark.html` uses its own fixed 14-model
  benchmark layout. The Web viewer loads Three.js from a CDN and is not a
  fully offline bundle.

All viewer/export paths share an important schema: accumulated 7-channel face
features plus 1-channel alpha are quantized into two RGBA8 LUTs, geometry uses
`alpha >= 0.5`, and 7 features plus 9 SH2 terms feed a fixed
`16 -> 16 -> 16 -> 3` color MLP. A checkpoint/schema change must be propagated
through `04_view_results.py`, `05_benchmark_fps.py`, `06_export_web.py`,
`py_viewer/scene.py`, the native shaders, and the Web shaders together.

Keep downloaded data in `datasets/`, training outputs in `results/`, exported
assets in `web/data/`, paper material in `paper/`, and documentation images in
`pics/`. `datasets/`, `results/`, `web/data/`, `tests/`, and `paper/` are all
ignored in this checkout. Do not infer that ignored local files are committed
or supported solely because they exist on disk.

## Training and Geometry-Prior Contracts

The common training path is:

1. load cameras/images and initialize independent triangles;
2. accumulate per-level feature/opacity buffers to the current lattice;
3. generate fragments, interpolate opacity, apply stochastic binary masking,
   and standard depth testing;
4. interpolate 7-D features, append 9-D SH2 view encoding, and shade with
   `ColorMLP`;
5. optimize a zero-valued opacity auxiliary surrogate plus edge-gradient
   surrogate and `0.8 * L1 + 0.2 * (1 - SSIM) / 2`.

Features and opacity begin at level 0. At step 5,000 the trainers preserve the
level-0 color field by lifting it to level 2, switch storage to levels `[2, 5]`,
and reinitialize opacity; do not describe opacity as preserved across the
lift. Topology is reconsidered every 100 steps, with visibility pruning and
screen-space edge splitting ending near step 9,500. Feature and opacity rows
must follow the returned parent/face map whenever topology changes.

`01_mip360.py` currently initializes and targets 15,000 faces regardless of
its exposed `n_points` argument. `02_mip360_test.py` is the variant in which
`--n_points` actually controls the target, using one-third random COLMAP points
and two-thirds farthest-point samples. It also keeps `--schedule_steps`
independent of a short `--steps` smoke run and refuses a nonempty output
directory unless `--overwrite` is explicit. `01_mip360.py` and
`03_random_init.py` do not provide that overwrite guard, so always use a fresh
ignored `--out_dir` for validation.

Geometry priors are active in `02_mip360_test.py` through fixed positive code
constants, so the current trainer loads both depth and normal data at startup.
There are no lambda/weight CLI options or compatibility aliases. Depth starts
at iteration 1 and decays over `schedule_steps`; normal starts at iteration
5,501, ramps for 500 steps, and both modalities sample 16,384 pixels per
selected view by default. The prior data contract is:

- `depth/<view-stem>.png`: uint16 encoded inverse camera-Z;
- `normals/` for full resolution or `normals_<downscale>/`: uint8 camera-space
  XYZ normals with `[127,127,127]` as the invalid sentinel;
- `sparse/0/depth_params.json`: per-view positive `png_scale`, finite offset,
  independent `depth_reliable` / `normal_reliable` flags, fit diagnostics, and
  the `camera_xyz_opencv_y_down` normal convention. The loader accepts legacy
  `scale` as a decode-scale alias, but scale magnitude is not a quality gate.

`08_prepare_geometry_priors.py` requires CUDA, binary COLMAP data, the pinned
ARAG source, and an external `ckpt_promask_best.pth`. It buffers coarse DA-v2
depth and Metric3D-v2 normals in CPU memory, refines them with ARAG, aligns
inverse depth to sparse COLMAP camera-Z, validates depth and normals
independently per view, writes neutral placeholders for unreliable modalities,
and records explicit reliability flags before publishing the three staged
targets with rollback. The scene is rejected only when no view provides either
a reliable depth or normal prior. It must not replace an existing prior layout
without `--overwrite`.

The training prior path deliberately fixes and detaches raster fragment
identity, raster-depth ordering, opacity, sampled pixels, cameras, and target
priors. Only ray-plane intersections and face normals recomputed from live
vertices are differentiable. Preserve that vertex-only contract, exact
front-to-back fixed-opacity compositing, face-forward camera-space normals,
and masked inverse-camera-Z loss unless a change is paired with focused
forward and finite-difference gradient tests. Depth supervision uses
`expected_camera_z / accumulated_opacity.detach()`, i.e. conditional expected
camera-Z, so sub-opaque coverage cannot move a single surface to
`z_target / opacity`. Normal supervision uses
`accumulated_opacity.detach() - dot(rendered_normal, prior_normal)`, the
opacity-weighted expectation of per-fragment angular error; this removes the
unoptimizable `1 - opacity` floor without opening an alpha gradient. Normal and
depth losses both exclude missing surfaces while retaining the full
sampled-row denominator. With the current fixed positive constants, the
experimental trainer loads both prior modalities and, at the first active step
and every 100 steps thereafter,
records separate hit fractions, accumulated-opacity q10/q50/q90 and fractions
above 0.5/0.9, normal concentration/cosine, and prior-specific vertex-gradient
norms. Depth supervision starts at iteration 1 and log-linearly decays from the
fixed `LAMBDA_DEPTH_PRIOR_INITIAL = 0.01` to
`LAMBDA_DEPTH_PRIOR_FINAL = 0.001` over `schedule_steps`. Normal supervision
uses the fixed `LAMBDA_NORMAL_PRIOR = 0.01`, remains disabled until
`normal_prior_start`, and then linearly ramps over `normal_prior_ramp_steps`.
The three lambda values are code constants, not CLI options; do not document
the removed `--lambda_*` options or the former `--*_prior_weight` aliases.

## Environment, Build, and Smoke Commands

The current README-tested reference is Ubuntu 22.04, Python 3.11, CUDA 13.0,
and an RTX 4070 Ti SUPER; `pyproject.toml` supports Python 3.10 or newer and
the documented PyTorch install command uses the cu124 wheel index. Always
report the actual OS, compiler, GPU, driver, CUDA toolkit, Python, and PyTorch
versions used for native, rendering, or performance validation rather than
assuming those reference values.

On Windows, verify `Get-Command python` and
`python -c "import sys; print(sys.executable)"` before running checks. In this
workspace the intended runtime is the `diffsoup` Conda environment; the base
`python` can resolve to a different Python without the required packages.
Build `fused-ssim` from an x64 Visual Studio developer prompt. A temporary
`TORCH_DONT_CHECK_COMPILER_ABI=1` was needed in one validated localized-MSVC
setup only after confirming ABI compatibility; never persist that override
globally.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
git submodule update --init --recursive
python -m pip install --no-build-isolation ./submodules/fused-ssim
python -m pip install -v .
python -m pip install -r requirements.txt
python examples/00_version.py
```

The root wheel metadata declares only PyTorch, while eager package imports also
need packages such as NumPy, SciPy, and imageio; install `requirements.txt`
before treating the full package/examples as runnable. `python -m pip install
-v .` builds and installs `_core` through scikit-build-core. Rebuild after
changing `src/`, `CMakeLists.txt`, or native compile definitions. The default
CUDA architecture is 89; set `CMAKE_CUDA_ARCHITECTURES` explicitly for another
target. Rebuild `fused-ssim` after changing its gitlink, PyTorch, CUDA, host
compiler, or target GPU.

Representative workflows are:

```bash
python examples/01_mip360.py --scene_root ./datasets/360_v2/garden --out_dir ./results/smoke_01_garden
python examples/03_random_init.py --scene lego --out_dir ./results/smoke_03_lego
python examples/08_prepare_geometry_priors.py --scene-root ./datasets/360_v2/garden
python examples/02_mip360_test.py --scene_root ./datasets/360_v2/garden --steps 20 --schedule_steps 10000 --out_dir ./results/smoke_02_garden
python examples/02_mip360_test.py --scene_root ./datasets/360_v2/garden --out_dir ./results/02_mip360/garden_arag
python examples/02_mip360_test_profile.py --scene_root ./datasets/360_v2/garden --out_dir ./results/profile_02_garden
python -m pip install -v viewer/
python examples/04_view_results.py --ckpt results/02_mip360/garden_arag/final_params.pt
```

For the Python viewer, install optional runtime dependencies separately:

```bash
python -m pip install glfw moderngl PyOpenGL imgui Pillow
python -m py_viewer.cli --ckpt results/02_mip360/garden_arag/final_params.pt --mode color
```

All tracked trainers and the root rasterizer are CUDA-required in practice even
where a script initially constructs a conditional `device`; CPU-only training
is not a supported fallback.

## Coding Style and Native/CUDA Conventions

Use four-space indentation. Python uses `snake_case` for functions/variables,
`PascalCase` for classes/dataclasses, and uppercase constants. Preserve type
hints and concise docstrings. Keep imports grouped, avoid unrelated mechanical
rewrites, and run `git diff --check` before handoff. There is no enforced
formatter or linter.

Validate tensor shape, dtype, device, contiguity, and empty-input behavior at
the public Python/native boundary. CUDA raster, multires, and encoding paths
expect contiguous float32 tensors and int32 indices on one CUDA device. Kernel
launches must use the supplied current PyTorch stream, with the nanobind device
guard selecting and restoring the input device. Avoid new default-stream work,
device-wide synchronization, hidden native `cudaMalloc`/`cudaFree`, redundant
zeroing, or workspaces that are unsafe across shapes/devices/streams. Existing
caller-stream synchronizations used to read exact fragment counts for
right-sized output allocation are intentional exceptions; do not add further
host-scalar synchronization without measurement and justification.

The CUDA custom-autograd boundaries are deliberate:

- `multires_triangle_color` differentiates feature tensors, not raster IDs or
  geometry;
- `edge_grad` is identity in forward and injects a silhouette gradient into
  projected positions in backward;
- `opacity_aux_loss` returns an exact zero value but carries an analytic alpha
  gradient, and cached fragments must remain equivalent to recomputation;
- `accumulate_to_level` builds an interpolation plan in forward and reuses it
  in backward. A target level may be above the stored range for feature lift.

CPU remeshing detaches tensors, converts them to CPU NumPy float32/int32,
executes native C++, and copies results back to the original device. It is a
synchronous, non-differentiable topology boundary. The clip splitter currently
considers an edge only when both endpoints lie inside the NDC cube; do not rely
on the broader wording in its Python docstring without reconciling the native
implementation.

Keep the shared SSIM contract unchanged: prediction first, target second, NCHW
values in `[0, 1]`, fused `padding="valid"`, and automatic
`pytorch_msssim(..., data_range=1.0)` fallback. The fused implementation only
differentiates its first argument, so determine its `train` flag from both
`prediction.requires_grad` and `torch.is_grad_enabled()`.

Checkpoint consumers require at least `V`, `F`, `feat_acc`, `alpha_acc`,
`color_mlp`, and `Rmax`. Keep `py_viewer/scene.py` on
`torch.load(..., weights_only=True)`. The native helper/export scripts currently
use `weights_only=False`; they may load only trusted local checkpoints and must
not broaden that unsafe boundary silently.

## Viewer and Rendering Conventions

Native and Python viewers require a real OpenGL 4.1 core context. The native
viewer build fetches GLFW, GLM, ImGui, and generates glad during CMake
configuration; it is independent of CUDA but not of the OpenGL/build toolchain.
Its benchmark opens a visible GLFW window and uses explicit GL completion for
timing; do not describe it as truly headless.

The native viewer currently hard-codes lattice level 5 and its Python/C++ API
does not receive checkpoint `Rmax`. Therefore use `04_view_results.py` and the
native `05_benchmark_fps.py` only with post-lift `Rmax=5` checkpoints until that
API is fixed. A short pre-step-5,000 checkpoint may be packed at `Rmax=0` but
sampled incorrectly by the native shader. `py_viewer` and Web carry the level
explicitly and do not share this limitation.

Preserve `py_viewer`'s geometry-pass MRT layout, deterministic alpha discard,
and default color output when adding debug modes. Depth means positive
camera-axis linear Z: `auto` changes only grayscale display contrast, while
`clip` uses the fixed near/far mapping. Normal output is flat world-space
`RGB = XYZ`: `face-forward` is view-dependent, while `oriented` preserves
triangle winding. PNG depth/normal screenshots are 8-bit visualizations, not
lossless floating-point exports.

Native/Python viewer defaults and Web-export background metadata are not always
the same. Normalize background, camera, resolution, quantization, and render
mode before comparing screenshots or hashes across surfaces.

## Testing, Profiling, and Validation

`tests/test_regularization.py` is the tracked expected-surface and
geometry-regularization regression file. There is no configured coverage
threshold. `.gitignore` explicitly unignores this tracked file while other
`tests/` contents remain ignored, so local tests can be invisible to ordinary
`rg --files`; inspect them with
`Get-ChildItem tests` or `rg --no-ignore --files tests`, and use
`git ls-files tests` to distinguish committed coverage from local-only files.
A new test intended for review must be deliberately force-added or unignored
and then confirmed in `git status`.

Ignored local tests are development-only and must not be treated as committed
coverage. The current local geometry-prior checks cover conditional depth,
missing-surface masking, fixed-fragment gradients, per-view reliability,
optional modality loading, schedule endpoints, removed CLI options, and COLMAP
half-pixel coordinates. Do not treat stale `.pyc` files as source. Run the
tracked test first, then any relevant ignored local checks that are present:

```bash
python -m pytest -q -p no:cacheprovider tests/test_regularization.py
# Additional ignored local checks, when present:
python -m pytest -q -p no:cacheprovider tests/test_cuda_optimizations.py
python -m pytest -q -p no:cacheprovider tests/test_geometry_priors.py
python -m compileall -q python py_viewer examples
pyrefly check --python-interpreter-path <activated-python> examples/utils.py examples/01_mip360.py examples/02_mip360_test.py examples/02_mip360_test_profile.py examples/03_random_init.py examples/08_prepare_geometry_priors.py py_viewer
python examples/00_version.py
```

Replace `<activated-python>` with the interpreter path verified above. Directly
checking the source `python/diffsoup` package currently reports the generated
`diffsoup._core` module as missing because no source stub is present; use
`compileall` plus the focused runtime tests for that package until a stub or
type-checker mapping is added.

Because depth is active from iteration 1, a short default `02_mip360_test.py`
run exercises the depth path but not the default normal start at iteration
5,501. For a joint wiring smoke test only, use a fresh ignored output directory
and pass `--normal_prior_start 1 --normal_prior_ramp_steps 0`; do not change the
fixed lambda constants merely to make a smoke run shorter.

CUDA changes must cover representative levels and feature dimensions, empty
geometry, exact/tolerance-based forward and backward parity, cached fragment
and accumulation-plan reuse, and non-default streams. SSIM changes require
forward and prediction-gradient parity against `pytorch_msssim` with valid
padding. Prior changes require focused tests for prior-file validation,
fixed-fragment compositing, invalid/empty samples, face-forward normals,
inverse-depth behavior, and finite-difference vertex gradients; reconcile the
ignored stale tests with the actual intended API before using them as gates.

For performance work, establish correctness first, warm up, use CUDA events or
explicit synchronization only around measurement, and report tensor shapes,
batch size, peak allocated memory, and full hardware/software details. For the
garden workload, benchmark forward plus backward at B1, B2, and B4. Windows
WDDM slowdown can appear only after the step-5,000 level lift, so run a separate
ignored garden B4 output through at least step 5,400 before claiming long-run
stability. `02_mip360_test_profile.py` disables tqdm output and writes
`train_profile.jsonl` with every-step wall time, sampled CUDA phase events, and
allocator snapshots at schedule boundaries. Keep profiling work in that
dedicated script so its synchronization and I/O cannot affect normal runs. Use
Nsight Compute on a small targeted workload for
occupancy, memory traffic, launch overhead, or initialization analysis.

Rendering changes require a real OpenGL context and a trusted representative
checkpoint, not shader-text inspection alone. Capture before/after images or
hashes and frame timings, and state whether native, Python, or Web rendering was
used. For native-viewer validation, also state that the checkpoint was
`Rmax=5` or explicitly account for the current fixed-level limitation.

## Commit and Pull Request Guidelines

Use short imperative subjects such as `Add ...`, `Fix ...`, `Optimize ...`, or
`Refine ...`, and keep each commit focused. Preserve unrelated user changes in
a dirty worktree. Pull requests should explain motivation and affected paths,
list exact validation commands and hardware/CUDA/OpenGL details, and include
metric deltas or screenshots for CUDA, viewer, Web, or rendering changes.

Do not commit datasets, checkpoints, result images, benchmark dumps, build
products, virtual environments, generated Web data, or ignored local paper/test
work unless the task explicitly changes that policy. When updating a submodule,
commit the intended gitlink and any necessary `.gitmodules` change together,
verify `git submodule status`, and keep submodule build, wheel, egg-info, model,
and cache artifacts untracked. `submodules/arag` intentionally uses
`ignore = untracked`; do not mistake its downloaded checkpoint or caches for a
gitlink change.
