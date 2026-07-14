# py_viewer/cli.py

from __future__ import annotations

import argparse
from pathlib import Path

from .scene import load_checkpoint_scene
from .viewer import DepthRange, NormalOrientation, RenderMode, launch_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the pure-Python DiffSoup OpenGL viewer."
    )
    parser.add_argument("--ckpt", required=True, help="Path to final_params.pt")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Screenshot directory (default: <checkpoint-dir>/py_viewer_output)",
    )
    parser.add_argument(
        "--up",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Override the checkpoint-derived world-up direction",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument(
        "--mode",
        choices=[mode.cli_name for mode in RenderMode],
        default=RenderMode.COLOR.cli_name,
        help="Initial render mode (default: color)",
    )
    parser.add_argument(
        "--depth-range",
        choices=[depth_range.value for depth_range in DepthRange],
        default=DepthRange.AUTO.value,
        help="Depth grayscale range: visible-scene auto contrast or fixed clips",
    )
    parser.add_argument(
        "--normal-orientation",
        choices=[orientation.value for orientation in NormalOrientation],
        default=NormalOrientation.FACE_FORWARD.value,
        help="Normal sign: face the camera or preserve triangle winding",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = Path(args.ckpt)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else checkpoint.parent / "py_viewer_output"
    )
    scene = load_checkpoint_scene(checkpoint, up=args.up)
    print(
        f"[py_viewer] {len(scene.verts):,} verts, {len(scene.faces):,} faces, "
        f"level={scene.level}, LUT={scene.lut0.shape[1]}x{scene.lut0.shape[0]}"
    )
    launch_scene(
        scene,
        output_dir=output_dir,
        width=max(1, args.width),
        height=max(1, args.height),
        render_mode=args.mode,
        depth_range=args.depth_range,
        normal_orientation=args.normal_orientation,
    )


if __name__ == "__main__":
    main()
