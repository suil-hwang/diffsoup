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


# ---------------------------------------------------------------------------
#  Validation helpers
# ---------------------------------------------------------------------------

def _normalized_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    """Return a positive integer ``(height, width)`` pair."""
    assert len(image_size) == 2 and not any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
        for value in image_size
    ), "image_size must be a positive integer pair"
    return int(image_size[0]), int(image_size[1])


# ---------------------------------------------------------------------------
#  Geometry-prior store
# ---------------------------------------------------------------------------

class GeometryPriorStore:
    """Load enabled scene PNG priors in training-view order."""

    def __init__(
        self,
        root: str | Path,
        view_names: Sequence[str],
        image_size: tuple[int, int],
        *,
        downscale: int,
        load_depth: bool = True,
        load_normal: bool = True,
    ) -> None:
        root = Path(root).resolve()
        self.view_names = tuple(view_names)
        self.image_size = _normalized_image_size(image_size)
        assert isinstance(load_depth, bool) and isinstance(load_normal, bool)
        assert load_depth or load_normal
        self.load_depth = load_depth
        self.load_normal = load_normal
        assert not isinstance(downscale, (bool, np.bool_))
        assert isinstance(downscale, (int, np.integer)) and downscale >= 0

        stems = tuple(Path(name).stem for name in self.view_names)
        assert len(stems) == len(set(stems)), "training view stems must be unique"

        depth_root = root / "depth"
        normal_folder = "normals" if downscale <= 1 else f"normals_{int(downscale)}"
        normal_root = root / normal_folder
        params_path = root / "sparse" / "0" / "depth_params.json"
        depth_params = {}
        if load_depth or params_path.is_file():
            depth_params = json.loads(params_path.read_text(encoding="utf-8"))
            assert isinstance(depth_params, dict), (
                f"{params_path}: expected an object keyed by view stem"
            )

        height, width = self.image_size
        count = len(self.view_names)
        self._depth_png = np.zeros((count, height, width), dtype=np.uint16)
        self._normal_rgb = np.full(
            (count, height, width, 3), 127, dtype=np.uint8,
        )
        self._depth_png_scale = np.ones(count, dtype=np.float32)
        self._depth_offset = np.zeros(count, dtype=np.float32)
        self._depth_reliable = np.zeros(count, dtype=np.bool_)
        self._normal_reliable = np.zeros(count, dtype=np.bool_)
        for row, stem in enumerate(stems):
            values = depth_params.get(stem, {})
            assert isinstance(values, dict), (
                f"{params_path}: {stem} parameters must be an object"
            )

            if load_depth:
                assert values, f"{params_path}: missing parameters for {stem}"
                scale_key = "png_scale" if "png_scale" in values else "scale"
                assert scale_key in values, f"{stem}: missing png_scale"
                png_scale = float(values[scale_key])
                depth_offset = float(values.get("offset", 0.0))
                with np.errstate(over="ignore", under="ignore"):
                    png_scale_f32 = np.float32(png_scale)
                    depth_offset_f32 = np.float32(depth_offset)
                assert np.isfinite(png_scale_f32) and png_scale_f32 > 0
                assert np.isfinite(depth_offset_f32)
                depth_reliable = values.get("depth_reliable", True)
                assert isinstance(depth_reliable, bool)

                depth_path = depth_root / f"{stem}.png"
                depth = iio.imread(depth_path)
                assert depth.shape == (height, width) and depth.dtype == np.uint16, (
                    f"{depth_path}: expected a uint16 {height}x{width} PNG"
                )
                self._depth_png[row] = depth
                self._depth_png_scale[row] = png_scale_f32
                self._depth_offset[row] = depth_offset_f32
                self._depth_reliable[row] = depth_reliable

            if load_normal:
                normal_reliable = values.get("normal_reliable", True)
                assert isinstance(normal_reliable, bool)
                convention = values.get(
                    "normal_convention", "camera_xyz_opencv_y_down",
                )
                assert convention == "camera_xyz_opencv_y_down", (
                    f"{stem}: unsupported normal_convention {convention!r}"
                )
                normal_path = normal_root / f"{stem}.png"
                normal = iio.imread(normal_path)
                assert (
                    normal.shape == (height, width, 3)
                    and normal.dtype == np.uint8
                ), f"{normal_path}: expected an RGB uint8 {height}x{width} PNG"
                self._normal_rgb[row] = normal
                self._normal_reliable[row] = normal_reliable

    def _view_indices_array(
        self,
        view_indices: Sequence[int] | torch.Tensor,
    ) -> np.ndarray:
        """Normalize requested training-view indices."""
        if isinstance(view_indices, torch.Tensor):
            raw = view_indices.detach().cpu().numpy()
        else:
            raw = np.asarray(view_indices)
        assert raw.ndim == 1 and not (
            raw.size and not np.issubdtype(raw.dtype, np.integer)
        ), "view_indices must be a one-dimensional integer sequence"
        indices = raw.astype(np.int64, copy=False)
        assert not ((indices < 0) | (indices >= len(self.view_names))).any()
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
        assert samples_per_view > 0
        assert dtype.is_floating_point
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
            inverse * self._depth_png_scale[indices, None]
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
        normal_valid = (
            self._normal_reliable[indices, None]
            & ~np.all(encoded_normal == 127, axis=-1)
        )
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
