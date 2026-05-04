"""Command line entry point for the Python-only DiffSoup viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .assets import list_exported_scenes, load_exported_scene


def configure_surface_format() -> None:
    from PyQt5 import QtGui

    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(0)
    fmt.setSwapInterval(0)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


def resolve_scene_dir(path: str | Path, model: str | None) -> Path:
    path = Path(path)
    if (path / "mesh.ply").exists():
        return path

    if model:
        candidate = path / model
        if (candidate / "mesh.ply").exists():
            return candidate
        raise FileNotFoundError(f"Model '{model}' not found under {path}")

    scenes = list_exported_scenes(path)
    if not scenes:
        raise FileNotFoundError(
            f"No exported DiffSoup scenes found under {path}"
        )
    lego = [p for p in scenes if p.name == "lego"]
    return lego[0] if lego else scenes[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Python-only viewer for DiffSoup web-exported assets.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Load a trained final_params.pt checkpoint directly.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="ours_mobile_results",
        help="Scene directory or root containing exported scene directories.",
    )
    parser.add_argument(
        "--model",
        default="lego",
        help="Model subdirectory to open when path is an asset root.",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for screenshots. Default: <scene>/py_viewer_output.",
    )
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument(
        "--up",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="World up direction for --ckpt. Auto-detected if omitted.",
    )
    args = parser.parse_args(argv)

    configure_surface_format()

    if args.ckpt:
        from .checkpoint import level_size, load_checkpoint_scene

        ckpt_path = Path(args.ckpt)
        output_dir = args.output_dir or str(ckpt_path.parent / "viewer_output")
        print(f"[py_viewer] loading checkpoint {ckpt_path}")
        scene = load_checkpoint_scene(ckpt_path, output_dir=output_dir, up=args.up)
        print(
            f"[py_viewer] {scene.name}: "
            f"{scene.verts.shape[0]:,} verts, {scene.faces.shape[0]:,} faces, "
            f"level={scene.level}, texels/face={level_size(scene.level)}, "
            f"lut={scene.lut0.shape[1]}x{scene.lut0.shape[0]}"
        )
    else:
        scene_dir = resolve_scene_dir(args.path, args.model)
        print(f"[py_viewer] loading {scene_dir}")
        scene = load_exported_scene(scene_dir)
        output_dir = args.output_dir
        print(
            f"[py_viewer] {scene.name}: "
            f"{scene.verts.shape[0]:,} verts, {scene.faces.shape[0]:,} faces, "
            f"level={scene.level}, lut={scene.lut0.shape[1]}x{scene.lut0.shape[0]}"
        )

    from PyQt5 import QtWidgets

    from .gl_viewer import DiffSoupViewerWindow

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    win = DiffSoupViewerWindow(scene, output_dir=output_dir)
    if args.fov is not None:
        win.controls.set_fov_value(args.fov)
    if args.near is not None:
        win.controls.set_near_value(args.near)
    if args.far is not None:
        win.controls.set_far_value(args.far)
    win.resize(max(args.width, 64), max(args.height, 64))
    win.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
