"""Native-compatible Python API for the Python-only DiffSoup viewer."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
import sys
import time
from typing import Any, Sequence, cast

import numpy as np
from PIL import Image
from PyQt5 import QtWidgets

from .assets import (
    SceneAssets,
    _array,
    scene_assets_from_arrays,
    scene_assets_from_split_luts,
)
from .cli import configure_surface_format
from .gl_viewer import DiffSoupGLWidget, DiffSoupViewerWindow

GL = cast(Any, import_module("OpenGL.GL"))


def launch_viewer(
    verts: np.ndarray,
    faces: np.ndarray,
    face_color_lut: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
    output_dir: str = "./results/viewer",
    up: Sequence[float] = (0, 0, 1),
) -> None:
    """Open an interactive viewer from in-memory numpy arrays.

    The signature matches the native ``diffsoupviewer.launch_viewer`` wrapper,
    so training/checkpoint code can pass vertices, faces, the packed float LUT,
    and MLP weights directly without first exporting ``mesh.ply`` and PNG/JSON
    files.
    """

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    scene = scene_assets_from_arrays(
        verts=verts,
        faces=faces,
        face_color_lut=face_color_lut,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W3=W3,
        b3=b3,
        output_dir=output_path,
        up=up,
    )

    configure_surface_format()
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    win = DiffSoupViewerWindow(scene, output_dir=output_path)
    win.resize(1200, 1200)
    win.show()
    app.exec_()


def _wait_for_widget_ready(app: QtWidgets.QApplication, widget: DiffSoupGLWidget) -> None:
    for _ in range(200):
        app.processEvents()
        if widget.is_ready:
            return
        time.sleep(0.01)
    raise RuntimeError("Timed out while initializing OpenGL benchmark widget.")


def _write_benchmark_outputs(output_dir: Path, times_ms: list[float]) -> None:
    frames_path = output_dir / "benchmark_frames.txt"
    with frames_path.open("w", encoding="utf-8") as f:
        for idx, elapsed in enumerate(times_ms):
            f.write(f"{idx} {elapsed}\n")

    mean = sum(times_ms) / len(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    summary_path = output_dir / "benchmark_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"frames: {len(times_ms)}\n")
        f.write(f"mean_ms: {mean}\n")
        f.write(f"min_ms:  {mn}\n")
        f.write(f"max_ms:  {mx}\n")
        f.write(f"fps:     {1000.0 / mean}\n")


def benchmark(
    verts: np.ndarray,
    faces: np.ndarray,
    lut0: np.ndarray,
    lut1: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
    mvps: np.ndarray,
    width: int = 1200,
    height: int = 1200,
    warmup: int = 10,
    save_every: int = 0,
    output_dir: str = "./results/viewer",
    up: Sequence[float] = (0, 0, 1),
    inv_mvps: np.ndarray | None = None,
) -> None:
    """Run the native-compatible fixed-MVP rendering benchmark.

    This mirrors ``diffsoupviewer.benchmark``: each MVP is rendered 100 times
    with GPU synchronization around the measured loop, per-frame timings are
    written to ``benchmark_frames.txt``, aggregate stats to
    ``benchmark_summary.txt``, and optional screenshots to
    ``screenshots/benchmark_%05d.png``. ``inv_mvps`` is an optional fast path
    for callers that can provide precomputed inverse MVPs in the same
    column-major layout as ``mvps``.
    """

    mvps_arr = _array("mvps", mvps, np.dtype(np.float32), (None, 4, 4))
    if mvps_arr.shape[0] == 0:
        raise ValueError("mvps must contain at least one matrix")
    # Native benchmark inputs are laid out for glm::make_mat4 (column-major).
    # Convert once before timing so Python transpose/allocation overhead is not
    # charged to every render_mvp_once call.
    render_mvps = np.ascontiguousarray(np.swapaxes(mvps_arr, 1, 2), dtype=np.float32)
    if inv_mvps is None:
        render_inv_mvps = np.ascontiguousarray(np.linalg.inv(render_mvps), dtype=np.float32)
    else:
        inv_mvps_arr = _array(
            "inv_mvps",
            inv_mvps,
            np.dtype(np.float32),
            (mvps_arr.shape[0], 4, 4),
        )
        render_inv_mvps = np.ascontiguousarray(
            np.swapaxes(inv_mvps_arr, 1, 2),
            dtype=np.float32,
        )

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    screenshots_dir = output_path / "screenshots"
    if save_every > 0:
        screenshots_dir.mkdir(parents=True, exist_ok=True)

    scene = scene_assets_from_split_luts(
        verts=verts,
        faces=faces,
        lut0=lut0,
        lut1=lut1,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W3=W3,
        b3=b3,
        output_dir=output_path,
        up=up,
    )

    configure_surface_format()
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    widget = DiffSoupGLWidget(scene, output_dir=output_path)
    widget.setWindowTitle("Benchmark")
    widget.resize(max(width, 1), max(height, 1))
    widget.set_auto_update(False)
    widget.show()
    _wait_for_widget_ready(app, widget)
    widget.set_auto_update(False)
    widget.set_render_size(max(width, 1), max(height, 1))

    final_target = widget.create_final_render_target(max(width, 1), max(height, 1))
    try:
        for _ in range(max(warmup, 0)):
            widget.render_mvp_once(
                render_mvps[0],
                inv_mvp=render_inv_mvps[0],
                target_fbo=final_target.fbo,
                blit_to_default=True,
            )
            app.processEvents()

        repeat = 100
        times_ms: list[float] = []
        for idx, render_mvp in enumerate(render_mvps):
            render_inv_mvp = render_inv_mvps[idx]
            GL.glFinish()
            t0 = time.perf_counter()
            for _ in range(repeat):
                widget.render_mvp_once(
                    render_mvp,
                    inv_mvp=render_inv_mvp,
                    target_fbo=final_target.fbo,
                    blit_to_default=True,
                )
            GL.glFinish()
            t1 = time.perf_counter()

            times_ms.append((t1 - t0) * 1000.0 / repeat)
            app.processEvents()

            if save_every > 0 and idx % save_every == 0:
                rgba = widget.read_current_rgba(
                    source_fbo=final_target.fbo,
                    width=final_target.width,
                    height=final_target.height,
                )
                path = screenshots_dir / f"benchmark_{idx:05d}.png"
                Image.fromarray(rgba, mode="RGBA").save(path)
                print(f"[py_viewer] saved {path}")

        _write_benchmark_outputs(output_path, times_ms)
    finally:
        widget.delete_final_render_target(final_target)
        widget.close()
        app.processEvents()
