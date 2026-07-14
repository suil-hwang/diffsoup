# DiffSoup Python Viewer

This directory reimplements the native `viewer/` package in Python while
preserving its rendering contract:

- a geometry pass interpolates the level-⁠`Rmax` triangle LUT and applies the
  deterministic `alpha >= 0.5` mask;
- a fullscreen pass reconstructs the view direction, evaluates SH2, and runs
  the trained `16 -> 16 -> 16 -> 3` color MLP;
- an orbit camera provides left-drag orbit, right/middle-drag pan, and wheel
  zoom; the settings panel controls color/depth/normal output, background, FOV,
  clipping, and screenshots.
- depth mode converts the OpenGL depth buffer to positive camera-axis linear
  depth. By default, a robust visible-scene range is mapped from black to white
  so nearby geometry is not compressed by a distant far clip; use
  `--depth-range clip` for a fixed near/far mapping. Normal mode displays
  flat world-space geometric normals as `RGB = XYZ`; its default
  `face-forward` orientation flips backfaces toward the current camera, while
  `--normal-orientation oriented` preserves the triangle winding sign.

## Install and Run

```bash
pip install numpy torch glfw PyOpenGL imgui Pillow
python -m py_viewer.cli \
  --ckpt results/02_synthetic/lego/final_params.pt \
  --mode normal \
  --normal-orientation face-forward
```

Use `--up 0 0 1` to override the checkpoint-derived world-up direction. Press
`1`, `2`, or `3` to select color, depth, or normal mode, `S` to save
`screenshot.png` in the output directory, and `Esc` to close.

The automatic depth range uses robust percentiles of in-frustum vertex depths
and may change as the camera moves. It changes only the grayscale display
mapping, not depth testing or geometry. Use `--depth-range clip` when images
must share a fixed metric scale. PNG screenshots are 8-bit visualizations, not
lossless floating-point depth exports.

Normal output is flat per triangle because DiffSoup checkpoints represent a
triangle soup and do not contain shared-vertex normals. `face-forward` is
useful for the viewer's two-sided rendering but is view-dependent. Use
`oriented` to diagnose winding or to compare stable world-space directions
across camera views. PNG normal screenshots use standard unsigned 8-bit
encoding `rgb = normal * 0.5 + 0.5`.

The public Python API mirrors the native wrapper at a higher level:

```python
from py_viewer import load_checkpoint_scene, launch_scene

scene = load_checkpoint_scene("results/02_synthetic/lego/final_params.pt")
launch_scene(scene, output_dir="results/python_viewer", render_mode="depth")
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
