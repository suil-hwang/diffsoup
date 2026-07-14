# Repository Guidelines

## Project Structure & Module Organization

`python/diffsoup/` is the public Python package; it wraps the compiled `_core` extension and contains rasterization, multiresolution, remeshing, point, and optimization APIs. Native C++17/CUDA code lives in `src/`, with kernels under `src/cuda/` and nanobind bindings in `src/main.cpp`. Runnable workflows are in `examples/` (`01_mip360.py`, `02_synthetic.py`, and `03_random_init.py` are the main training entry points). `viewer/` is a separately installable native OpenGL viewer, while `web/` contains the WebGL viewer and mobile benchmark. Documentation images belong in `pics/`. Keep downloaded data and generated outputs in ignored paths such as `datasets/`, `results/`, and `web/data/`.

## Build, Test, and Development Commands

- `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124` installs the tested CUDA 12.4 PyTorch build.
- `pip3 install -v .` builds and installs the CUDA/nanobind `diffsoup` extension through scikit-build-core.
- `pip3 install -r requirements.txt` installs example and evaluation dependencies.
- `python3 examples/00_version.py` verifies that the package and native extension import correctly.
- `python3 examples/03_random_init.py --scene lego` runs a representative training workflow.
- `pip3 install -v viewer/` builds the optional OpenGL viewer independently of CUDA.

The documented reference environment is Ubuntu 22.04, Python 3.10, CUDA 12.4, and an RTX 4090. Rebuild after modifying `src/` or `CMakeLists.txt`.

## Coding Style & Naming Conventions

Use four-space indentation. Python follows `snake_case` for functions and variables, `PascalCase` for classes/dataclasses, and uppercase names for constants. Preserve type hints, concise docstrings, and explicit tensor shape/dtype/device checks near CUDA boundaries. C++/CUDA uses C++17, descriptive lower-snake-case functions, and namespaces such as `diffsoup::cuda`. No formatter or linter is currently enforced; match the surrounding file and keep imports grouped.

## Testing Guidelines

There is currently no committed automated test suite or coverage threshold. At minimum, run the import smoke test and the affected example. Kernel changes should be rebuilt and checked on CUDA with representative tensor shapes; rendering changes should include before/after metrics or images. New Python tests should use `pytest` conventions under `tests/test_<feature>.py`.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects such as `Add ...`, `Fix ...`, and `Refine ...`; keep each commit focused. Pull requests should explain the motivation and affected paths, list exact validation commands and hardware/CUDA details, link relevant issues, and include screenshots or metric deltas for viewer, web, or rendering changes. Do not commit datasets, checkpoints, build products, or virtual environments.
