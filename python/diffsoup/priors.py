# python/diffsoup/priors.py
"""Load aligned scene-level depth and normal priors."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import imageio.v3 as iio
import numpy as np
import torch


class JointGeometryPriorSamples(NamedTuple):
    """Depth and normal samples sharing ``(batch, y, x)`` rows."""

    pixels_b_y_x: torch.Tensor
    inverse_camera_z: torch.Tensor
    depth_valid: torch.Tensor
    normal_camera: torch.Tensor
    normal_valid: torch.Tensor


def _normalized_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    """Return a positive integer ``(height, width)`` pair."""
    if len(image_size) != 2 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
        for value in image_size
    ):
        raise ValueError("image_size must be a positive integer pair")
    return int(image_size[0]), int(image_size[1])


class GeometryPriorStore:
    """Load standard scene PNG priors in training-view order."""

    def __init__(
        self,
        root: str | Path,
        view_names: Sequence[str],
        image_size: tuple[int, int],
        *,
        downscale: int,
    ) -> None:
        root = Path(root).resolve()
        self.view_names = tuple(view_names)
        self.image_size = _normalized_image_size(image_size)
        if (
            isinstance(downscale, (bool, np.bool_))
            or not isinstance(downscale, (int, np.integer))
            or downscale < 0
        ):
            raise ValueError("downscale must be a nonnegative integer")

        stems = tuple(Path(name).stem for name in self.view_names)
        if len(stems) != len(set(stems)):
            raise ValueError("requested training view stems are not unique")

        depth_root = root / "depth"
        normal_folder = "normals" if downscale <= 1 else f"normals_{int(downscale)}"
        normal_root = root / normal_folder
        params_path = root / "sparse" / "0" / "depth_params.json"
        depth_params = json.loads(params_path.read_text(encoding="utf-8"))
        if not isinstance(depth_params, dict):
            raise ValueError(f"{params_path}: expected an object keyed by view stem")

        try:
            all_scales = np.asarray(
                [values["scale"] for values in depth_params.values()],
                dtype=np.float64,
            )
            affine = np.asarray(
                [
                    (
                        depth_params[stem]["scale"],
                        depth_params[stem].get("offset", 0.0),
                    )
                    for stem in stems
                ],
                dtype=np.float64,
            ).reshape(-1, 2)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{params_path}: invalid depth scale parameters") from error

        positive_scales = all_scales[np.isfinite(all_scales) & (all_scales > 0)]
        if positive_scales.size == 0 and stems:
            raise ValueError(f"{params_path}: no finite positive depth scales")
        median_scale = (
            float(np.median(positive_scales)) if positive_scales.size else 0.0
        )

        with np.errstate(over="ignore", under="ignore"):
            depth_scale = affine[:, 0].astype(np.float32)
            depth_offset = affine[:, 1].astype(np.float32)
        valid_affine = (
            np.isfinite(depth_scale)
            & (depth_scale > 0)
            & np.isfinite(depth_offset)
        )
        if not valid_affine.all():
            row = int(np.flatnonzero(~valid_affine)[0])
            raise ValueError(f"{stems[row]}: invalid depth scale or offset")

        height, width = self.image_size
        count = len(self.view_names)
        self._depth_png = np.empty((count, height, width), dtype=np.uint16)
        self._normal_rgb = np.empty((count, height, width, 3), dtype=np.uint8)
        self._depth_scale = depth_scale
        self._depth_offset = depth_offset
        self._depth_reliable = (
            (affine[:, 0] >= 0.2 * median_scale)
            & (affine[:, 0] <= 5.0 * median_scale)
        )
        for row, stem in enumerate(stems):
            depth_path = depth_root / f"{stem}.png"
            normal_path = normal_root / f"{stem}.png"

            depth = iio.imread(depth_path)
            if depth.shape != (height, width) or depth.dtype != np.uint16:
                raise ValueError(
                    f"{depth_path}: expected a uint16 {height}x{width} PNG"
                )
            normal = iio.imread(normal_path)
            if normal.shape != (height, width, 3) or normal.dtype != np.uint8:
                raise ValueError(
                    f"{normal_path}: expected an RGB uint8 {height}x{width} PNG"
                )
            self._depth_png[row] = depth
            self._normal_rgb[row] = normal

    def _view_indices_array(
        self,
        view_indices: Sequence[int] | torch.Tensor,
    ) -> np.ndarray:
        """Normalize requested training-view indices."""
        if isinstance(view_indices, torch.Tensor):
            raw = view_indices.detach().cpu().numpy()
        else:
            raw = np.asarray(view_indices)
        if raw.ndim != 1 or (
            raw.size and not np.issubdtype(raw.dtype, np.integer)
        ):
            raise TypeError("view_indices must be a one-dimensional integer sequence")
        indices = raw.astype(np.int64, copy=False)
        if ((indices < 0) | (indices >= len(self.view_names))).any():
            raise IndexError("view index is outside the training-view range")
        return indices

    def sample_joint_uniform(
        self,
        view_indices: Sequence[int] | torch.Tensor,
        samples_per_view: int,
        rng: np.random.Generator,
        device: torch.device | str,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> JointGeometryPriorSamples:
        """Sample aligned depth and normal at uniform full-image pixels."""
        if samples_per_view <= 0:
            raise ValueError("samples_per_view must be positive")
        if not dtype.is_floating_point:
            raise TypeError("prior output dtype must be floating point")
        indices = self._view_indices_array(view_indices)
        batch_size = indices.size

        height, width = self.image_size
        flat = rng.integers(
            0,
            height * width,
            size=(batch_size, samples_per_view),
            dtype=np.int64,
        )
        y = flat // width
        x = flat - y * width
        rows = indices[:, None]

        encoded_depth = self._depth_png[rows, y, x]
        inverse = encoded_depth.astype(np.float32) / 65536.0
        inverse = (
            inverse * self._depth_scale[indices, None]
            + self._depth_offset[indices, None]
        )
        depth_valid = (
            self._depth_reliable[indices, None]
            & (encoded_depth > 0)
            & np.isfinite(inverse)
            & (inverse > 0)
        )
        inverse[~depth_valid] = 0.0

        encoded_normal = self._normal_rgb[rows, y, x]
        normal_valid = ~np.all(encoded_normal == 127, axis=-1)
        normal = encoded_normal.astype(np.float32) / 255.0 * 2.0 - 1.0
        length = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal /= np.maximum(length, 1e-8)
        normal[~normal_valid] = 0.0

        local_batch = np.repeat(
            np.arange(batch_size, dtype=np.int64), samples_per_view,
        )
        pixels = np.column_stack((local_batch, y.ravel(), x.ravel()))
        return JointGeometryPriorSamples(
            torch.from_numpy(pixels).to(device=device),
            torch.from_numpy(inverse.reshape(-1)).to(device=device, dtype=dtype),
            torch.from_numpy(depth_valid.reshape(-1)).to(device=device),
            torch.from_numpy(normal.reshape(-1, 3)).to(device=device, dtype=dtype),
            torch.from_numpy(normal_valid.reshape(-1)).to(device=device),
        )
