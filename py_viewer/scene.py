# py_viewer/scene.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def level_size(level: int) -> int:
    """Return the number of LUT samples per face at one subdivision level."""
    if level < 0:
        raise ValueError("level must be non-negative")
    if level == 0:
        return 3
    return ((1 << (level - 1)) + 1) * ((1 << level) + 1)


def _as_numpy(value, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def pack_face_color_lut(
    feat_acc,
    alpha_acc,
    num_faces: int,
    level: int,
) -> np.ndarray:
    """Pack seven features and opacity into a padded ``[H, W, 8]`` LUT."""
    features = _as_numpy(feat_acc, np.float32)
    alpha = _as_numpy(alpha_acc, np.float32)
    if features.ndim == 3:
        features = features.reshape(-1, features.shape[-1])
    if alpha.ndim == 3:
        alpha = alpha.reshape(-1, alpha.shape[-1])
    if features.ndim != 2 or features.shape[1] != 7:
        raise ValueError(f"expected feature shape [N, 7], got {features.shape}")
    if alpha.ndim != 2 or alpha.shape[1] != 1:
        raise ValueError(f"expected alpha shape [N, 1], got {alpha.shape}")

    sample_count = num_faces * level_size(level)
    if sample_count <= 0:
        raise ValueError("a scene must contain at least one face")
    if features.shape[0] < sample_count or alpha.shape[0] < sample_count:
        raise ValueError(
            "checkpoint LUT is smaller than num_faces * level_size(level)"
        )

    flat = np.concatenate(
        [features[:sample_count], alpha[:sample_count]], axis=-1
    )
    width = min(4096, sample_count)
    height = (sample_count + width - 1) // width
    padded = np.zeros((height * width, 8), dtype=np.float32)
    padded[:sample_count] = flat
    return padded.reshape(height, width, 8)


def extract_mlp_weights(state_dict: Mapping[str, object]) -> tuple[np.ndarray, ...]:
    """Extract the three linear layers from a ``ColorMLP`` state dictionary."""
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for name, value in state_dict.items():
        if name.endswith("weight"):
            weights.append(_as_numpy(value, np.float32))
        elif name.endswith("bias"):
            biases.append(_as_numpy(value, np.float32))
    if len(weights) != 3 or len(biases) != 3:
        raise ValueError(
            f"expected three MLP layers, found {len(weights)} weights and "
            f"{len(biases)} biases"
        )
    expected = ((16, 16), (16, 16), (3, 16), (16,), (16,), (3,))
    arrays = (*weights, *biases)
    if tuple(array.shape for array in arrays) != expected:
        raise ValueError(
            "unsupported ColorMLP architecture: "
            f"{tuple(array.shape for array in arrays)}"
        )
    w1, w2, w3, b1, b2, b3 = arrays
    return w1, b1, w2, b2, w3, b3


def detect_up(checkpoint: Mapping[str, object]) -> tuple[float, float, float]:
    """Infer the world-up direction using the native viewer convention."""
    if "up" in checkpoint:
        up = checkpoint["up"]
        if hasattr(up, "detach"):
            up = up.detach().cpu().tolist()
        return float(up[0]), float(up[1]), float(up[2])
    if "flip_z" in checkpoint:
        return 0.0, -1.0, 0.0
    return 0.0, 0.0, 1.0


@dataclass
class SceneData:
    """Validated GPU upload payload for one DiffSoup scene."""

    verts: np.ndarray
    faces: np.ndarray
    lut0: np.ndarray
    lut1: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray
    level: int = 5
    up: Sequence[float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        self.verts = _as_numpy(self.verts, np.float32)
        self.faces = _as_numpy(self.faces, np.int32)
        self.lut0 = _as_numpy(self.lut0, np.uint8)
        self.lut1 = _as_numpy(self.lut1, np.uint8)
        self.w1 = _as_numpy(self.w1, np.float32)
        self.b1 = _as_numpy(self.b1, np.float32)
        self.w2 = _as_numpy(self.w2, np.float32)
        self.b2 = _as_numpy(self.b2, np.float32)
        self.w3 = _as_numpy(self.w3, np.float32)
        self.b3 = _as_numpy(self.b3, np.float32)
        self.level = int(self.level)

        if self.verts.ndim != 2 or self.verts.shape[1] != 3 or not len(self.verts):
            raise ValueError("verts must be a non-empty float32 [V, 3] array")
        if not np.isfinite(self.verts).all():
            raise ValueError("verts must contain only finite values")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or not len(self.faces):
            raise ValueError("faces must be a non-empty int32 [F, 3] array")
        if self.faces.min() < 0 or self.faces.max() >= len(self.verts):
            raise ValueError("faces contain out-of-range vertex indices")
        if self.lut0.shape != self.lut1.shape or self.lut0.ndim != 3:
            raise ValueError("lut0 and lut1 must have identical [H, W, 4] shapes")
        if self.lut0.shape[2] != 4:
            raise ValueError("LUT textures must have four channels")
        if self.level < 0:
            raise ValueError("level must be non-negative")
        required_samples = len(self.faces) * level_size(self.level)
        if self.lut0.shape[0] * self.lut0.shape[1] < required_samples:
            raise ValueError("LUT capacity is smaller than faces * level_size(level)")

        expected_shapes = {
            "w1": (16, 16), "b1": (16,),
            "w2": (16, 16), "b2": (16,),
            "w3": (3, 16), "b3": (3,),
        }
        for name, shape in expected_shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")

        up = np.asarray(self.up, dtype=np.float32)
        if up.shape != (3,) or not np.isfinite(up).all():
            raise ValueError("up must contain three finite values")
        length = float(np.linalg.norm(up))
        if length <= 1e-8:
            raise ValueError("up must be non-zero")
        self.up = tuple(float(v) for v in up / length)

    @property
    def center(self) -> np.ndarray:
        return self.verts.mean(axis=0, dtype=np.float64).astype(np.float32)

    @classmethod
    def from_face_color_lut(
        cls,
        verts,
        faces,
        face_color_lut,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        *,
        level: int = 5,
        up: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> "SceneData":
        lut = _as_numpy(face_color_lut, np.float32)
        if lut.ndim != 3 or lut.shape[2] != 8:
            raise ValueError("face_color_lut must have shape [H, W, 8]")
        lut_u8 = np.clip(lut * 255.0, 0.0, 255.0).astype(np.uint8)
        return cls(
            verts, faces, lut_u8[..., :4], lut_u8[..., 4:],
            w1, b1, w2, b2, w3, b3, level=level, up=up,
        )


def load_checkpoint_scene(
    path: str | Path,
    *,
    up: Sequence[float] | None = None,
) -> SceneData:
    """Load ``final_params.pt`` and convert it to the viewer upload format."""
    import torch

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    required = {"V", "F", "feat_acc", "alpha_acc", "color_mlp", "Rmax"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise KeyError(f"checkpoint is missing: {', '.join(missing)}")

    verts = _as_numpy(checkpoint["V"], np.float32)
    faces = _as_numpy(checkpoint["F"], np.int32)
    level = int(checkpoint["Rmax"])
    lut = pack_face_color_lut(
        checkpoint["feat_acc"], checkpoint["alpha_acc"], len(faces), level
    )
    w1, b1, w2, b2, w3, b3 = extract_mlp_weights(checkpoint["color_mlp"])
    return SceneData.from_face_color_lut(
        verts,
        faces,
        lut,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        level=level,
        up=up if up is not None else detect_up(checkpoint),
    )
