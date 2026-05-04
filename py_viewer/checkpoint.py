"""Load trained DiffSoup checkpoints into Python viewer assets."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .assets import SceneAssets, scene_assets_from_arrays


def level_size(level: int) -> int:
    """Number of LUT texels per face at subdivision ``level``."""

    if level == 0:
        return 3
    a = (1 << (level - 1)) + 1
    b = (1 << level) + 1
    return a * b


def _to_numpy(value, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def pack_face_color_lut(
    feat_acc: np.ndarray,
    alpha_acc: np.ndarray,
    num_faces: int,
    level: int,
) -> np.ndarray:
    """Pack checkpoint feature and alpha accumulators into ``[H, W, 8]``."""

    feat = np.asarray(feat_acc, dtype=np.float32)
    alpha = np.asarray(alpha_acc, dtype=np.float32)
    if feat.ndim == 3:
        feat = feat.reshape(-1, feat.shape[-1])
    if alpha.ndim == 3:
        alpha = alpha.reshape(-1, alpha.shape[-1])

    expected = num_faces * level_size(level)
    if feat.shape[0] < expected or alpha.shape[0] < expected:
        raise ValueError(
            "feat_acc/alpha_acc do not contain enough texels: "
            f"need {expected}, got {feat.shape[0]} and {alpha.shape[0]}"
        )
    if feat.shape[-1] != 7:
        raise ValueError(f"expected feat_dim=7, got {feat.shape[-1]}")
    if alpha.shape[-1] != 1:
        raise ValueError(f"expected alpha dim=1, got {alpha.shape[-1]}")

    lut_flat = np.concatenate([feat[:expected], alpha[:expected]], axis=-1)
    tex_width = min(4096, expected)
    tex_height = math.ceil(expected / tex_width)
    padded = np.zeros((tex_height * tex_width, 8), dtype=np.float32)
    padded[:expected] = lut_flat
    return np.ascontiguousarray(padded.reshape(tex_height, tex_width, 8))


def extract_mlp_weights(state_dict: Mapping[str, object]) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Pull W1, b1, W2, b2, W3, b3 from a ColorMLP state dict."""

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for key, value in state_dict.items():
        arr = _to_numpy(value, np.dtype(np.float32))
        if "weight" in key:
            weights.append(arr)
        elif "bias" in key:
            biases.append(arr)

    if len(weights) < 3 or len(biases) < 3:
        raise ValueError(
            f"Expected at least 3 linear layers, found "
            f"{len(weights)} weights / {len(biases)} biases. "
            f"Keys: {list(state_dict.keys())}"
        )

    W1, W2, W3 = weights[0], weights[1], weights[2]
    b1, b2, b3 = biases[0], biases[1], biases[2]
    if W1.shape != (16, 16) or W2.shape != (16, 16) or W3.shape != (3, 16):
        raise ValueError(f"Unexpected MLP weight shapes: {W1.shape}, {W2.shape}, {W3.shape}")
    if b1.shape != (16,) or b2.shape != (16,) or b3.shape != (3,):
        raise ValueError(f"Unexpected MLP bias shapes: {b1.shape}, {b2.shape}, {b3.shape}")
    return (
        np.ascontiguousarray(W1),
        np.ascontiguousarray(b1),
        np.ascontiguousarray(W2),
        np.ascontiguousarray(b2),
        np.ascontiguousarray(W3),
        np.ascontiguousarray(b3),
    )


def detect_up(ckpt: Mapping[str, object]) -> tuple[float, float, float]:
    """Infer world-up from checkpoint metadata."""

    if "up" in ckpt:
        up = ckpt["up"]
        arr = _to_numpy(up, np.dtype(np.float32)).reshape(3)
        return (float(arr[0]), float(arr[1]), float(arr[2]))
    if "flip_z" in ckpt:
        return (0.0, -1.0, 0.0)
    return (0.0, 0.0, 1.0)


def load_checkpoint_scene(
    ckpt_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    up: Sequence[float] | None = None,
) -> SceneAssets:
    """Load ``final_params.pt`` into ``SceneAssets`` without exporting files."""

    import torch

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = ["V", "F", "feat_acc", "alpha_acc", "Rmax", "color_mlp"]
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint missing required keys: {', '.join(missing)}")

    verts = _to_numpy(ckpt["V"], np.dtype(np.float32))
    faces = _to_numpy(ckpt["F"], np.dtype(np.int32))
    feat_acc = _to_numpy(ckpt["feat_acc"], np.dtype(np.float32))
    alpha_acc = _to_numpy(ckpt["alpha_acc"], np.dtype(np.float32))
    level = int(ckpt["Rmax"])

    face_color_lut = pack_face_color_lut(
        feat_acc=feat_acc,
        alpha_acc=alpha_acc,
        num_faces=faces.shape[0],
        level=level,
    )
    W1, b1, W2, b2, W3, b3 = extract_mlp_weights(ckpt["color_mlp"])
    resolved_up = tuple(up) if up is not None else detect_up(ckpt)
    resolved_output_dir = Path(output_dir) if output_dir is not None else path.parent / "viewer_output"

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
        output_dir=resolved_output_dir,
        up=resolved_up,
        level=level,
    )
    return SceneAssets(
        scene_dir=scene.scene_dir,
        name=path.parent.name or path.stem,
        verts=scene.verts,
        faces=scene.faces,
        lut0=scene.lut0,
        lut1=scene.lut1,
        W1=scene.W1,
        b1=scene.b1,
        W2=scene.W2,
        b2=scene.b2,
        W3=scene.W3,
        b3=scene.b3,
        level=scene.level,
        up=scene.up,
        background=scene.background,
    )
