"""Load DiffSoup web-exported scene assets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SceneAssets:
    """In-memory representation of one exported DiffSoup scene."""

    scene_dir: Path
    name: str
    verts: np.ndarray
    faces: np.ndarray
    lut0: np.ndarray
    lut1: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W3: np.ndarray
    b3: np.ndarray
    level: int
    up: np.ndarray
    background: np.ndarray


def _array(name: str, value, dtype: np.dtype, shape: tuple[int | None, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {arr.ndim}")
    for axis, expected in enumerate(shape):
        if expected is not None and arr.shape[axis] != expected:
            raise ValueError(
                f"{name} shape mismatch at axis {axis}: "
                f"expected {expected}, got {arr.shape[axis]}"
            )
    return np.ascontiguousarray(arr, dtype=dtype)


def scene_assets_from_arrays(
    verts: np.ndarray,
    faces: np.ndarray,
    face_color_lut: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    W3: np.ndarray,
    b3: np.ndarray,
    output_dir: str | Path = "./results/viewer",
    up: Sequence[float] = (0, 0, 1),
    level: int = 5,
) -> SceneAssets:
    """Pack native ``diffsoupviewer.launch_viewer`` inputs into ``SceneAssets``."""

    verts_arr = _array("verts", verts, np.dtype(np.float32), (None, 3))
    faces_arr = _array("faces", faces, np.dtype(np.int32), (None, 3))
    lut_arr = _array("face_color_lut", face_color_lut, np.dtype(np.float32), (None, None, 8))

    clipped = np.clip(lut_arr, 0.0, 1.0)
    lut0 = np.ascontiguousarray((clipped[..., :4] * 255).astype(np.uint8))
    lut1 = np.ascontiguousarray((clipped[..., 4:] * 255).astype(np.uint8))

    return SceneAssets(
        scene_dir=Path(output_dir),
        name="arrays",
        verts=verts_arr,
        faces=faces_arr,
        lut0=lut0,
        lut1=lut1,
        W1=_array("W1", W1, np.dtype(np.float32), (16, 16)),
        b1=_array("b1", b1, np.dtype(np.float32), (16,)),
        W2=_array("W2", W2, np.dtype(np.float32), (16, 16)),
        b2=_array("b2", b2, np.dtype(np.float32), (16,)),
        W3=_array("W3", W3, np.dtype(np.float32), (3, 16)),
        b3=_array("b3", b3, np.dtype(np.float32), (3,)),
        level=level,
        up=_array("up", np.asarray(up), np.dtype(np.float32), (3,)),
        background=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )


def scene_assets_from_split_luts(
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
    output_dir: str | Path = "./results/viewer",
    up: Sequence[float] = (0, 0, 1),
    level: int = 5,
) -> SceneAssets:
    """Pack native ``diffsoupviewer.benchmark`` split-LUT inputs."""

    verts_arr = _array("verts", verts, np.dtype(np.float32), (None, 3))
    faces_arr = _array("faces", faces, np.dtype(np.int32), (None, 3))
    lut0_arr = _array("lut0", lut0, np.dtype(np.uint8), (None, None, 4))
    lut1_arr = _array("lut1", lut1, np.dtype(np.uint8), (None, None, 4))
    if lut0_arr.shape != lut1_arr.shape:
        raise ValueError(
            f"lut0 and lut1 sizes must match, got {lut0_arr.shape} and {lut1_arr.shape}"
        )

    return SceneAssets(
        scene_dir=Path(output_dir),
        name="arrays",
        verts=verts_arr,
        faces=faces_arr,
        lut0=lut0_arr,
        lut1=lut1_arr,
        W1=_array("W1", W1, np.dtype(np.float32), (16, 16)),
        b1=_array("b1", b1, np.dtype(np.float32), (16,)),
        W2=_array("W2", W2, np.dtype(np.float32), (16, 16)),
        b2=_array("b2", b2, np.dtype(np.float32), (16,)),
        W3=_array("W3", W3, np.dtype(np.float32), (3, 16)),
        b3=_array("b3", b3, np.dtype(np.float32), (3,)),
        level=level,
        up=_array("up", np.asarray(up), np.dtype(np.float32), (3,)),
        background=np.array([1.0, 1.0, 1.0], dtype=np.float32),
    )


def list_exported_scenes(root: str | Path) -> list[Path]:
    """Return subdirectories that look like web-exported DiffSoup scenes."""

    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_dir()
        and (p / "mesh.ply").exists()
        and (p / "lut0.png").exists()
        and (p / "lut1.png").exists()
        and (p / "mlp_weights.json").exists()
    )


def load_exported_scene(scene_dir: str | Path) -> SceneAssets:
    """Load one ``06_export_web.py`` style asset directory."""

    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    required = ["mesh.ply", "lut0.png", "lut1.png", "mlp_weights.json"]
    missing = [name for name in required if not (scene_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Scene directory is missing required files: {', '.join(missing)}"
        )

    meta = _load_meta(scene_dir / "meta.json")
    verts, faces = load_binary_triangle_ply(scene_dir / "mesh.ply")
    lut0 = _load_rgba(scene_dir / "lut0.png")
    lut1 = _load_rgba(scene_dir / "lut1.png")
    weights = _load_mlp(scene_dir / "mlp_weights.json")

    return SceneAssets(
        scene_dir=scene_dir,
        name=scene_dir.name,
        verts=verts,
        faces=faces,
        lut0=lut0,
        lut1=lut1,
        level=int(meta.get("level", 5)),
        up=np.asarray(meta.get("up", [0.0, 0.0, 1.0]), dtype=np.float32),
        background=np.asarray(meta.get("background", [1.0, 1.0, 1.0]), dtype=np.float32),
        **weights,
    )


def load_binary_triangle_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the binary little-endian triangle PLY written by ``06_export_web.py``."""

    path = Path(path)
    with path.open("rb") as f:
        header = _read_ply_header(f)
        if header.get("format") != "binary_little_endian":
            raise ValueError(f"Only binary_little_endian PLY is supported: {path}")

        num_verts = int(header["vertex"])
        num_faces = int(header["face"])

        verts = np.frombuffer(f.read(num_verts * 3 * 4), dtype="<f4")
        if verts.size != num_verts * 3:
            raise ValueError(f"Unexpected EOF while reading vertices: {path}")
        verts = verts.reshape(num_verts, 3).astype(np.float32, copy=False)

        face_dtype = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
        faces_raw = np.frombuffer(f.read(num_faces * face_dtype.itemsize), dtype=face_dtype)
        if faces_raw.size != num_faces:
            raise ValueError(f"Unexpected EOF while reading faces: {path}")
        if not np.all(faces_raw["n"] == 3):
            raise ValueError("Only triangular PLY faces are supported.")
        faces = faces_raw["idx"].astype(np.int32, copy=True)

    return np.ascontiguousarray(verts), np.ascontiguousarray(faces)


def _read_ply_header(f) -> dict[str, str | int]:
    first = f.readline().decode("ascii", errors="strict").strip()
    if first != "ply":
        raise ValueError("Not a PLY file.")

    header: dict[str, str | int] = {}
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF in PLY header.")
        text = line.decode("ascii", errors="strict").strip()
        if text == "end_header":
            break
        parts = text.split()
        if not parts:
            continue
        if parts[0] == "format":
            header["format"] = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            header[parts[1]] = int(parts[2])

    if "vertex" not in header or "face" not in header:
        raise ValueError("PLY header must contain vertex and face elements.")
    return header


def _load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    return np.ascontiguousarray(arr)


def _load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_mlp(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    def arr(name: str, shape: Iterable[int]) -> np.ndarray:
        out = np.asarray(data[name], dtype=np.float32).reshape(tuple(shape))
        return np.ascontiguousarray(out)

    return {
        "W1": arr("W1", (16, 16)),
        "b1": arr("b1", (16,)),
        "W2": arr("W2", (16, 16)),
        "b2": arr("b2", (16,)),
        "W3": arr("W3", (3, 16)),
        "b3": arr("b3", (3,)),
    }
