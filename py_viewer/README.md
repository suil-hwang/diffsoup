# Python DiffSoup Viewer

Python-only viewer for DiffSoup web-exported assets. It avoids the native
`diffsoupviewer` C++/nanobind extension, but still renders on the GPU through
PyQt5 and PyOpenGL.

## Run

From the repository root:

```powershell
pip install -e ".[py-viewer]"
```

For lower PyOpenGL wrapper overhead, install the accelerated extra instead:

```powershell
pip install -e ".[py-viewer-fast]"
```

Then run:

```powershell
python -m py_viewer ours_mobile_results --model lego
```

You can also pass one scene directory directly:

```powershell
python -m py_viewer ours_mobile_results\chair
```

You can open a trained checkpoint directly:

```powershell
python -m py_viewer --ckpt results\02_synthetic\lego\final_params.pt
```

Use `--up X Y Z` to override the auto-detected world-up direction.

## Native-compatible API

`py_viewer.launch_viewer` accepts the same in-memory arrays as the native
`diffsoupviewer.launch_viewer` wrapper:

```python
import py_viewer

py_viewer.launch_viewer(
    verts=verts,
    faces=faces,
    face_color_lut=face_color_lut,  # float32 [H, W, 8]
    W1=W1, b1=b1,
    W2=W2, b2=b2,
    W3=W3, b3=b3,
    output_dir="./results/viewer",
    up=(0, 0, 1),
)
```

`py_viewer.benchmark` mirrors the native benchmark API:

```python
py_viewer.benchmark(
    verts=verts,
    faces=faces,
    lut0=lut0, lut1=lut1,  # uint8 [H, W, 4]
    W1=W1, b1=b1,
    W2=W2, b2=b2,
    W3=W3, b3=b3,
    mvps=mvps,             # float32 [B, 4, 4], native column-major layout
    inv_mvps=inv_mvps,     # optional inverse MVPs in the same layout
    width=1200,
    height=1200,
    warmup=10,
    save_every=0,
    output_dir="./results/viewer",
    up=(0, 0, 1),
)
```

Benchmark outputs match the native viewer naming:

- `benchmark_frames.txt`
- `benchmark_summary.txt`
- `screenshots/benchmark_%05d.png` when `save_every > 0`

## Controls

- Floating Settings panel: RGB background inputs, FOV slider, logarithmic near/far clip sliders, reset, screenshot
- Left drag: orbit
- Right drag or Shift+drag: pan
- Wheel: zoom
- `R`: reset view
- `S`: save screenshot to `<scene>/py_viewer_output`
- `Esc`: close

## Input Format

The viewer expects the files produced by `examples/06_export_web.py`:

- `mesh.ply`
- `lut0.png`
- `lut1.png`
- `mlp_weights.json`
- `meta.json` optional, but recommended
