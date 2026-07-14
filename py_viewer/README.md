# DiffSoup Python Viewer

This directory reimplements the native `viewer/` package in Python while
preserving its rendering contract:

- a geometry pass interpolates the level-⁠`Rmax` triangle LUT and applies the
  deterministic `alpha >= 0.5` mask;
- a fullscreen pass reconstructs the view direction, evaluates SH2, and runs
  the trained `16 -> 16 -> 16 -> 3` color MLP;
- an orbit camera provides left-drag orbit, right/middle-drag pan, and wheel
  zoom; the settings panel controls background, FOV, clipping, and screenshots.

## Install and Run

```bash
pip install numpy torch glfw PyOpenGL imgui Pillow
python -m py_viewer.cli \
  --ckpt results/02_synthetic/lego/final_params.pt
```

Use `--up 0 0 1` to override the checkpoint-derived world-up direction. Press
`S` to save `screenshot.png` in the output directory and `Esc` to close.

The public Python API mirrors the native wrapper at a higher level:

```python
from py_viewer import load_checkpoint_scene, launch_scene

scene = load_checkpoint_scene("results/02_synthetic/lego/final_params.pt")
launch_scene(scene, output_dir="results/python_viewer")
```

`py_viewer.benchmark(...)` accepts the same mesh, LUT, MLP, and
column-major MVP payload used by `diffsoupviewer.benchmark`. It writes
`benchmark_frames.txt`, `benchmark_summary.txt`, and optional screenshots.

## Native-to-Python Mapping

| Native implementation                   | Python implementation           |
| --------------------------------------- | ------------------------------- |
| `viewer/src/camera.*`                   | `py_viewer/camera.py`  |
| GLSL in `viewer/src/viewer.cpp`         | `py_viewer/*.glsl`     |
| OpenGL loop and benchmark               | `py_viewer/viewer.py`  |
| nanobind wrapper + `04_view_results.py` | `scene.py`, `cli.py`            |

The Python version favors inspectability over minimum CPU overhead. Rendering
remains GPU-backed and uses the same RGBA8 LUT quantization as the native and
web viewers.
