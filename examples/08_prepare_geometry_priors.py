"""Prepare ARAG depth and normal supervision for a MipNeRF-360 scene."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator, Sequence

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from utils import (
    _colmap_root,
    _image_folder_name,
    _intrinsics_from_camera,
    _qvec2rotmat,
    _read_cameras,
)


_MASK64 = (1 << 64) - 1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARAG_ROOT = _REPOSITORY_ROOT / "submodules" / "arag"
_DEFAULT_CHECKPOINT = _ARAG_ROOT / "work_dir" / "ckpts" / "ckpt_promask_best.pth"
_ARAG_INFER_MODULE = "_diffsoup_arag_infer"
_ARAG_MODULE_ROOTS = ("xformers", "src", "hubconf", "mono")

_MIN_DEPTH_VALID_FRACTION = 0.5
_MIN_NORMAL_VALID_FRACTION = 0.5
_MAX_DEPTH_MEDIAN_ABS_REL = 0.35
_MAX_DEPTH_P90_ABS_REL = 1.0
_MIN_SPARSE_FIT_POINTS = 32
_MIN_SPARSE_VALIDATION_POINTS = 8


def _canonicalize_arag_normals(
    raw_normal: torch.Tensor,
    K: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ARAG normals to unit face-forward camera-space XYZ."""
    if raw_normal.ndim != 3 or raw_normal.shape[-1] != 3:
        raise ValueError("raw_normal must have shape (H, W, 3)")
    if not raw_normal.is_floating_point():
        raise TypeError("raw_normal must be floating point")
    if K.shape != (3, 3):
        raise ValueError("K must have shape (3, 3)")
    normal = raw_normal.detach()
    height, width, _ = normal.shape
    K = K.detach().to(device=normal.device, dtype=normal.dtype)
    focal = K[(0, 1), (0, 1)]
    if not torch.isfinite(K).all() or (focal.abs() <= eps).any():
        raise ValueError("K must be finite with nonzero focal lengths")

    length = torch.linalg.vector_norm(normal, dim=-1)
    valid = torch.isfinite(normal).all(dim=-1) & (length > eps)
    normal = F.normalize(
        torch.where(valid.unsqueeze(-1), normal, torch.zeros_like(normal)),
        dim=-1,
        eps=eps,
    )
    y = torch.arange(height, dtype=normal.dtype, device=normal.device) + 0.5
    x = torch.arange(width, dtype=normal.dtype, device=normal.device) + 0.5
    pixel_y, pixel_x = torch.meshgrid(y, x, indexing="ij")
    direction = torch.stack(
        (
            (pixel_x - K[0, 2]) / K[0, 0],
            (pixel_y - K[1, 2]) / K[1, 1],
            torch.ones_like(pixel_x),
        ),
        dim=-1,
    )
    normal = torch.where(
        ((normal * direction).sum(dim=-1) > 0).unsqueeze(-1), -normal, normal,
    )
    normal = torch.where(valid.unsqueeze(-1), normal, torch.zeros_like(normal))
    return normal, valid


def fit_inverse_depth_affine(
    relative_depth: np.ndarray,
    target_inverse_camera_z: np.ndarray,
) -> tuple[float, float]:
    """Fit ``target = slope * relative + shift`` with Huber IRLS."""
    relative = np.asarray(relative_depth, dtype=np.float64).reshape(-1)
    target = np.asarray(target_inverse_camera_z, dtype=np.float64).reshape(-1)
    if relative.shape != target.shape:
        raise ValueError("relative and target depth arrays must have equal size")
    valid = np.isfinite(relative) & np.isfinite(target) & (target > 0)
    relative = relative[valid]
    target = target[valid]
    if relative.size < 2 or np.ptp(relative) <= np.finfo(np.float64).eps:
        raise ValueError("at least two nonconstant valid depth samples are required")

    design = np.stack((relative, np.ones_like(relative)), axis=-1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    for _ in range(20):
        residual = design @ solution - target
        center = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - center))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            break
        threshold = 1.345 * scale
        absolute = np.abs(residual - center)
        weight = np.ones_like(absolute)
        outside = absolute > threshold
        weight[outside] = threshold / absolute[outside]
        root_weight = np.sqrt(weight)
        updated, *_ = np.linalg.lstsq(
            design * root_weight[:, None], target * root_weight, rcond=None,
        )
        if np.allclose(updated, solution, rtol=1e-10, atol=1e-12):
            solution = updated
            break
        solution = updated
    slope, shift = map(float, solution)
    if not np.isfinite(slope) or not np.isfinite(shift):
        raise ValueError("inverse-depth affine fit produced nonfinite values")
    return slope, shift


def _arag_module_name(name: str) -> bool:
    return (
        name == _ARAG_INFER_MODULE
        or name.partition(".")[0] in _ARAG_MODULE_ROOTS
    )


@contextmanager
def _arag_inference_module() -> Iterator[ModuleType]:
    """Load bundled ARAG helpers without retaining global import changes."""
    original_path = sys.path.copy()
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if _arag_module_name(name)
    }
    infer_path = _ARAG_ROOT / "tools" / "infer.py"
    try:
        for name in tuple(sys.modules):
            if _arag_module_name(name):
                sys.modules.pop(name)
        sys.modules["xformers"] = None
        sys.modules["xformers.ops"] = None
        spec = importlib.util.spec_from_file_location(
            _ARAG_INFER_MODULE, infer_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load ARAG inference module from {infer_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_ARAG_INFER_MODULE] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = original_path
        for name in tuple(sys.modules):
            if _arag_module_name(name):
                sys.modules.pop(name)
        sys.modules.update(original_modules)


@dataclass(frozen=True)
class ColmapImage:
    """COLMAP image record including sparse 2D-to-3D observations."""

    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    points_xy: np.ndarray
    point3d_ids: np.ndarray


@dataclass(frozen=True)
class SceneData:
    image_folder: str
    image_size: tuple[int, int]
    records: tuple[ColmapImage, ...]
    image_paths: tuple[Path, ...]
    cameras: dict[int, dict]
    points: dict[int, np.ndarray]


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"unexpected end of COLMAP file (wanted {size} bytes)")
    return data


def _unpack(handle, fmt: str) -> tuple:
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))


def read_colmap_images_bin(path: str | Path) -> list[ColmapImage]:
    """Read COLMAP images.bin without discarding observations."""
    records: list[ColmapImage] = []
    with Path(path).open("rb") as handle:
        (count,) = _unpack(handle, "<Q")
        for _ in range(count):
            _unpack(handle, "<I")  # image_id
            qvec = np.asarray(_unpack(handle, "<dddd"), dtype=np.float64)
            tvec = np.asarray(_unpack(handle, "<ddd"), dtype=np.float64)
            (camera_id,) = _unpack(handle, "<I")
            name_bytes = bytearray()
            while (byte := _read_exact(handle, 1)) != b"\x00":
                name_bytes.extend(byte)
            (num_points,) = _unpack(handle, "<Q")
            points_xy = np.empty((num_points, 2), dtype=np.float64)
            point3d_ids = np.empty(num_points, dtype=np.int64)
            for index in range(num_points):
                x, y, point_id = _unpack(handle, "<ddq")
                points_xy[index] = (x, y)
                point3d_ids[index] = point_id
            records.append(ColmapImage(
                qvec, tvec, camera_id, name_bytes.decode("utf-8"),
                points_xy, point3d_ids,
            ))
    return records


def read_colmap_points3d_bin(path: str | Path) -> dict[int, np.ndarray]:
    """Read point IDs and XYZ positions from COLMAP points3D.bin."""
    points: dict[int, np.ndarray] = {}
    with Path(path).open("rb") as handle:
        (count,) = _unpack(handle, "<Q")
        for _ in range(count):
            (point_id,) = _unpack(handle, "<Q")
            xyz = np.asarray(_unpack(handle, "<ddd"), dtype=np.float64)
            _read_exact(handle, 3)
            _unpack(handle, "<d")
            (track_length,) = _unpack(handle, "<Q")
            _read_exact(handle, track_length * 8)
            points[point_id] = xyz
    return points


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _fit_point_mask(point_ids: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (_splitmix64(int(point_id)) % 10 < 8 for point_id in point_ids),
        dtype=np.bool_, count=point_ids.size,
    )


def _ordered_records(records: Sequence[ColmapImage]) -> list[ColmapImage]:
    """Sort all COLMAP views and verify scene-layout-safe names."""
    ordered = sorted(records, key=lambda record: record.name)
    stems = [Path(record.name).stem for record in ordered]
    if len(stems) != len(set(stems)):
        raise ValueError("selected image stems are not unique")
    return ordered


def _resolve_image(scene_root: Path, folder: str, name: str) -> Path:
    path = scene_root / folder / name
    if not path.is_file():
        path = scene_root / folder / Path(name).name
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {name}")
    return path


def _load_scene(scene_root: Path, downscale: int) -> SceneData:
    sparse_root = Path(_colmap_root(str(scene_root)))
    images_path = sparse_root / "images.bin"
    points_path = sparse_root / "points3D.bin"
    # A previous fixed-layout publish may have created sparse/0 beside a
    # flattened source model. Keep reading that original source model.
    flattened = scene_root / "sparse"
    if (
        not (images_path.is_file() and points_path.is_file())
        and (flattened / "images.bin").is_file()
        and (flattened / "points3D.bin").is_file()
    ):
        sparse_root = flattened
        images_path = sparse_root / "images.bin"
        points_path = sparse_root / "points3D.bin"
    if not images_path.is_file() or not points_path.is_file():
        raise FileNotFoundError("ARAG preparation requires COLMAP images.bin and points3D.bin")
    records = tuple(_ordered_records(read_colmap_images_bin(images_path)))
    if not records:
        raise ValueError("COLMAP model contains no images")
    image_folder = _image_folder_name(downscale)
    image_paths = tuple(
        _resolve_image(scene_root, image_folder, record.name) for record in records
    )
    image_size = tuple(int(v) for v in iio.improps(image_paths[0]).shape[:2])
    for path in image_paths[1:]:
        size = tuple(int(v) for v in iio.improps(path).shape[:2])
        if size != image_size:
            raise ValueError(f"all scene images must share one size; {path} is {size}")
    cameras = _read_cameras(str(sparse_root))
    missing_camera = next((r.camera_id for r in records if r.camera_id not in cameras), None)
    if missing_camera is not None:
        raise KeyError(f"COLMAP camera {missing_camera} is missing")
    return SceneData(
        image_folder, image_size, records, image_paths, cameras,
        read_colmap_points3d_bin(points_path),
    )


def _scaled_intrinsics(camera: dict, height: int, width: int) -> np.ndarray:
    K, original_height, original_width = _intrinsics_from_camera(camera)
    result = K.numpy().astype(np.float64)
    result[0, :] *= width / float(original_width)
    result[1, :] *= height / float(original_height)
    return result


def _normalize_coarse_pair(
    depth: np.ndarray,
    normal: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one coarse depth-normal pair for in-memory refinement."""
    depth = np.asarray(depth)
    normal = np.asarray(normal)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if normal.shape == (3,) + image_size:
        normal = np.moveaxis(normal, 0, -1)
    expected_normal = image_size + (3,)
    if depth.shape != image_size or normal.shape != expected_normal:
        raise RuntimeError(
            "unexpected coarse output shapes: "
            f"depth={depth.shape}, normal={normal.shape}"
        )
    return (
        np.ascontiguousarray(depth, dtype=np.float32),
        np.ascontiguousarray(normal, dtype=np.float32),
    )


def _sample_bilinear(image: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape
    x, y = xy[:, 0], xy[:, 1]
    inside = (
        np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= width - 1)
        & (y >= 0) & (y <= height - 1)
    )
    result = np.full(x.shape, np.nan, dtype=np.float64)
    if not inside.any():
        return result, inside
    xi, yi = x[inside], y[inside]
    x0, y0 = np.floor(xi).astype(np.int64), np.floor(yi).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = xi - x0, yi - y0
    result[inside] = (
        image[y0, x0] * (1 - wx) * (1 - wy)
        + image[y0, x1] * wx * (1 - wy)
        + image[y1, x0] * (1 - wx) * wy
        + image[y1, x1] * wx * wy
    )
    return result, inside


def _sparse_samples(
    record: ColmapImage,
    camera: dict,
    points: dict[int, np.ndarray],
    raw_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = record.point3d_ids >= 0
    ids, xy = record.point3d_ids[observed], record.points_xy[observed]
    present = np.fromiter(
        (int(point_id) in points for point_id in ids),
        dtype=np.bool_, count=ids.size,
    )
    ids, xy = ids[present], xy[present]
    if ids.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), np.empty(0, dtype=np.int64)
    xyz = np.stack([points[int(point_id)] for point_id in ids])
    rotation = _qvec2rotmat(record.qvec)
    camera_z = xyz @ rotation[2, :] + record.tvec[2]
    height, width = raw_depth.shape
    scaled_xy = np.empty_like(xy)
    scaled_xy[:, 0] = (xy[:, 0] + 0.5) * width / float(camera["w"]) - 0.5
    scaled_xy[:, 1] = (xy[:, 1] + 0.5) * height / float(camera["h"]) - 0.5
    relative, inside = _sample_bilinear(raw_depth, scaled_xy)
    valid = inside & np.isfinite(relative) & np.isfinite(camera_z) & (camera_z > 0)
    return relative[valid], 1.0 / camera_z[valid], ids[valid]


def _transform_depth(
    values: np.ndarray,
    *,
    reciprocal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if not reciprocal:
        return values, np.isfinite(values)
    valid = np.isfinite(values) & (values > 1e-8)
    transformed = np.zeros_like(values)
    np.divide(1.0, values, out=transformed, where=valid)
    return transformed, valid


def _depth_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    relative = np.abs(prediction - target) / np.maximum(target, 1e-12)
    return {
        "mean_abs_rel": float(np.mean(relative)),
        "median_abs_rel": float(np.median(relative)),
        "p90_abs_rel": float(np.percentile(relative, 90)),
    }


def _fit_depth(relative: np.ndarray, target: np.ndarray, point_ids: np.ndarray) -> dict:
    fit_partition = _fit_point_mask(point_ids)
    successes, errors = [], []
    for transform, reciprocal in (("identity", False), ("reciprocal", True)):
        transformed, transform_valid = _transform_depth(
            relative, reciprocal=reciprocal,
        )
        common = transform_valid & np.isfinite(target) & (target > 0)
        fit, validation = common & fit_partition, common & ~fit_partition
        fit_count = int(fit.sum())
        validation_count = int(validation.sum())
        if fit_count < _MIN_SPARSE_FIT_POINTS:
            errors.append(f"{transform}: only {fit_count} fit points")
            continue
        if validation_count < _MIN_SPARSE_VALIDATION_POINTS:
            errors.append(
                f"{transform}: only {validation_count} validation points"
            )
            continue
        fit_values = transformed[fit]
        if np.ptp(fit_values) <= np.finfo(np.float64).eps:
            errors.append(f"{transform}: fit samples are constant")
            continue
        slope, shift = fit_inverse_depth_affine(fit_values, target[fit])
        if slope <= 0:
            errors.append(f"{transform}: affine slope is not positive")
            continue
        successes.append({
            "transform": transform,
            "slope": slope,
            "shift": shift,
            "validation_metrics": _depth_metrics(
                slope * transformed[validation] + shift, target[validation],
            ),
        })
    if not successes:
        raise ValueError("no valid depth transform; " + "; ".join(errors))
    return min(
        successes,
        # Prefer a publishable transform before ranking its residual quality.
        key=lambda item: (
            item["validation_metrics"]["median_abs_rel"]
            > _MAX_DEPTH_MEDIAN_ABS_REL
            or item["validation_metrics"]["p90_abs_rel"]
            > _MAX_DEPTH_P90_ABS_REL,
            item["validation_metrics"]["median_abs_rel"],
            item["validation_metrics"]["mean_abs_rel"],
        ),
    )


def _canonical_depth(raw: np.ndarray, fit: dict) -> tuple[np.ndarray, np.ndarray]:
    transformed, raw_valid = _transform_depth(
        raw, reciprocal=fit["transform"] == "reciprocal",
    )
    canonical = fit["slope"] * transformed + fit["shift"]
    valid = raw_valid & np.isfinite(canonical) & (canonical > 0)
    return np.where(valid, canonical, 0.0).astype(np.float32), valid


def _encode_depth_png(
    inverse_depth: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, float]:
    encoded = np.zeros(inverse_depth.shape, dtype=np.uint16)
    high = float(inverse_depth[valid].max())
    encoded[valid] = np.rint(
        np.clip(inverse_depth[valid] / high, 0.0, 1.0) * 65535.0
    ).astype(np.uint16)
    scale = high * 65536.0 / 65535.0
    return encoded, scale


def _encode_normal_png(
    normal: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    """Encode canonical unit XYZ normals and neutral invalid pixels."""
    safe = np.where(valid[..., None], normal, 0.0)
    return np.clip((safe + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)


def _ceil_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _infer_coarse(
    infer_module: ModuleType,
    scene: SceneData,
) -> deque[tuple[np.ndarray, np.ndarray]]:
    """Generate DA-v2 depth and Metric3D-v2 normals in CPU memory."""
    import cv2
    from tqdm.auto import tqdm

    height, width = scene.image_size
    gib = len(scene.records) * height * width * 16 / float(1 << 30)
    print(f"[coarse] buffering {len(scene.records)} views ({gib:.2f} GiB)")
    coarse: deque[tuple[np.ndarray, np.ndarray]] = deque()
    dav2_model = metric3d_model = None
    try:
        dav2_model = infer_module.load_dav2_model("vitl", "cuda")
        metric3d_model = infer_module.load_metric3d_model("ViT-Small", "cuda")
        for image_path in tqdm(scene.image_paths, desc="coarse priors"):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"OpenCV failed to decode {image_path}")
            depth = infer_module.run_dav2(dav2_model, image_bgr, input_size=518)
            normal = infer_module.run_metric3d(
                metric3d_model, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                device="cuda",
            )
            coarse.append(_normalize_coarse_pair(
                depth, normal, scene.image_size,
            ))
    finally:
        if dav2_model is not None:
            del dav2_model
        if metric3d_model is not None:
            del metric3d_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return coarse


def _run_arag_refiner(
    model,
    image_bgr: np.ndarray,
    coarse_depth: np.ndarray,
    coarse_normal: np.ndarray,
    aligned_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run URGT and return depth plus camera-space XYZ normal arrays."""
    import cv2

    original_height, original_width = image_bgr.shape[:2]
    height, width = aligned_size
    pad_bottom, pad_right = height - original_height, width - original_width
    if pad_bottom or pad_right:
        border = (0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
        image_bgr = cv2.copyMakeBorder(image_bgr, *border)
        coarse_depth = cv2.copyMakeBorder(coarse_depth, *border)
        coarse_normal = cv2.copyMakeBorder(coarse_normal, *border)

    coarse_depth = coarse_depth / (float(coarse_depth.max()) + 1e-8)
    coarse_normal = coarse_normal / (
        np.linalg.norm(coarse_normal, axis=-1, keepdims=True) + 1e-12
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_t = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
    depth_t = torch.from_numpy(coarse_depth).float()[None].cuda()
    normal_t = torch.from_numpy(coarse_normal.astype(np.float32)).permute(2, 0, 1)[None].cuda()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, log_dict = model(
            image_highres=image_t.half(),
            coarse_depth=depth_t.half(),
            coarse_normal=normal_t.half(),
            depth_valid_mask=torch.ones_like(depth_t).half(),
            normal_valid_mask=torch.ones_like(normal_t).half(),
        )
    depth = log_dict["depth_pred"].float()[..., :original_height, :original_width]
    # URGT preserves the Metric3D camera-space XYZ channel order.
    normal = log_dict["normal_pred"].squeeze(0).float()
    normal = normal[:, :original_height, :original_width]
    depth_np = depth.squeeze().cpu().numpy().astype(np.float32)
    normal_np = normal.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    expected_depth = (original_height, original_width)
    expected_normal = expected_depth + (3,)
    if depth_np.shape != expected_depth or normal_np.shape != expected_normal:
        raise RuntimeError(
            "unexpected ARAG output shapes: "
            f"depth={depth_np.shape}, normal={normal_np.shape}"
        )
    return depth_np, normal_np


def _validate_frame(
    name: str,
    fit: dict,
    depth_valid: np.ndarray,
    normal_valid: np.ndarray,
) -> None:
    failures = []
    if float(depth_valid.mean()) < _MIN_DEPTH_VALID_FRACTION:
        failures.append("insufficient valid depth")
    if float(normal_valid.mean()) < _MIN_NORMAL_VALID_FRACTION:
        failures.append("insufficient valid normals")
    metrics = fit["validation_metrics"]
    if metrics["median_abs_rel"] > _MAX_DEPTH_MEDIAN_ABS_REL:
        failures.append("depth median AbsRel is too large")
    if metrics["p90_abs_rel"] > _MAX_DEPTH_P90_ABS_REL:
        failures.append("depth p90 AbsRel is too large")
    if failures:
        raise ValueError(f"{name}: " + "; ".join(failures))


def _stage_refined_priors(
    infer_module: ModuleType,
    scene: SceneData,
    coarse_priors: deque[tuple[np.ndarray, np.ndarray]],
    staging_root: Path,
    normal_folder: str,
    checkpoint: Path,
    patch_split: tuple[int, int],
) -> dict:
    """Refine, validate, and directly encode every view into staging."""
    import cv2
    from tqdm.auto import tqdm

    depth_root = staging_root / "depth"
    normal_root = staging_root / normal_folder
    depth_root.mkdir(parents=True)
    normal_root.mkdir(parents=True)
    height, width = scene.image_size
    aligned_size = (
        _ceil_multiple(height, patch_split[0]),
        _ceil_multiple(width, patch_split[1]),
    )
    model = None
    depth_params: dict[str, dict[str, float]] = {}
    try:
        processor, built_height, built_width = infer_module.build_patch_processor(
            aligned_size[0], aligned_size[1], patch_split,
        )
        if (built_height, built_width) != aligned_size:
            raise RuntimeError("ARAG patch processor changed padded size")
        model = infer_module.build_urgt_model(
            str(checkpoint), processor, min_depth=1e-3,
            max_depth=80.0, device="cuda",
        )
        progress = tqdm(
            zip(scene.records, scene.image_paths),
            total=len(scene.records),
            desc="ARAG priors",
        )
        for record, image_path in progress:
            stem = Path(record.name).stem
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"OpenCV failed to decode {image_path}")
            coarse_depth, coarse_normal = coarse_priors.popleft()
            raw_depth, raw_normal = _run_arag_refiner(
                model, image_bgr, coarse_depth, coarse_normal, aligned_size,
            )
            del coarse_depth, coarse_normal

            camera = scene.cameras[record.camera_id]
            relative, target, point_ids = _sparse_samples(
                record, camera, scene.points, raw_depth,
            )
            fit = _fit_depth(relative, target, point_ids)
            inverse_depth, depth_valid = _canonical_depth(raw_depth, fit)
            K = _scaled_intrinsics(camera, height, width)
            normal_t, normal_valid_t = _canonicalize_arag_normals(
                torch.from_numpy(raw_normal),
                torch.from_numpy(K.astype(np.float32)),
            )
            normal, normal_valid = normal_t.numpy(), normal_valid_t.numpy()
            _validate_frame(record.name, fit, depth_valid, normal_valid)
            depth_u16, scale = _encode_depth_png(inverse_depth, depth_valid)
            normal_u8 = _encode_normal_png(normal, normal_valid)
            iio.imwrite(depth_root / f"{stem}.png", depth_u16)
            iio.imwrite(normal_root / f"{stem}.png", normal_u8)
            depth_params[stem] = {"scale": scale, "offset": 0.0}
            progress.set_postfix(
                transform=fit["transform"],
                med=f"{fit['validation_metrics']['median_abs_rel']:.3f}",
            )
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    params_path = staging_root / "sparse" / "0" / "depth_params.json"
    params_path.parent.mkdir(parents=True)
    params_path.write_text(
        json.dumps(depth_params, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"frames": len(scene.records)}


def _scene_prior_targets(scene_root: Path, normal_folder: str) -> dict[str, Path]:
    return {
        "depth": scene_root / "depth",
        "normal": scene_root / normal_folder,
        "params": scene_root / "sparse" / "0" / "depth_params.json",
    }


def _ensure_publishable(scene_root: Path, normal_folder: str, overwrite: bool) -> None:
    if overwrite:
        return
    for path in _scene_prior_targets(scene_root, normal_folder).values():
        if path.exists():
            raise FileExistsError(
                f"scene prior target already exists ({path}); pass --overwrite"
            )


def _publish_scene_priors(
    scene_root: str | Path,
    staging_root: str | Path,
    normal_folder: str,
    overwrite: bool,
) -> None:
    """Publish a validated staged layout with params as the commit marker."""
    scene_root = Path(scene_root).resolve()
    staging_root = Path(staging_root).resolve()
    if not normal_folder or Path(normal_folder).name != normal_folder:
        raise ValueError("normal_folder must be one directory name")
    final = _scene_prior_targets(scene_root, normal_folder)
    staged = _scene_prior_targets(staging_root, normal_folder)
    if not all((
        staged["depth"].is_dir(),
        staged["normal"].is_dir(),
        staged["params"].is_file(),
    )):
        raise FileNotFoundError("staging layout is incomplete")
    _ensure_publishable(scene_root, normal_folder, overwrite)

    backup_root = staging_root / ".publish-backup"
    if backup_root.exists():
        raise FileExistsError(backup_root)
    backup = _scene_prior_targets(backup_root, normal_folder)
    moved_old: set[str] = set()
    moved_new: set[str] = set()
    try:
        # Removing params first makes every intermediate overwrite state fail closed.
        for key in ("params", "depth", "normal"):
            if final[key].exists():
                backup[key].parent.mkdir(parents=True, exist_ok=True)
                os.replace(final[key], backup[key])
                moved_old.add(key)
        for key in ("depth", "normal", "params"):
            final[key].parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[key], final[key])
            moved_new.add(key)
    except BaseException as publish_error:
        rollback_failed = False
        for key in ("params", "normal", "depth"):
            if key in moved_new and final[key].exists():
                try:
                    staged[key].parent.mkdir(parents=True, exist_ok=True)
                    os.replace(final[key], staged[key])
                except OSError:
                    rollback_failed = True
        # Restore data directories before restoring the old commit marker.
        for key in ("depth", "normal"):
            if key in moved_old and backup[key].exists():
                try:
                    final[key].parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup[key], final[key])
                except OSError:
                    rollback_failed = True
        if not rollback_failed and "params" in moved_old and backup["params"].exists():
            try:
                os.replace(backup["params"], final["params"])
            except OSError:
                rollback_failed = True
        if rollback_failed:
            raise RuntimeError(
                f"scene prior publish and rollback both failed; backups remain at {backup_root}"
            ) from publish_error
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    shutil.rmtree(backup_root, ignore_errors=True)


def prepare_arag_scene(args: argparse.Namespace) -> dict:
    """Run ARAG end to end and publish the standard scene layout."""
    scene_root = Path(args.scene_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not scene_root.is_dir():
        raise FileNotFoundError(scene_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.downscale < 0:
        raise ValueError("downscale must be nonnegative")
    patch_split = tuple(int(value) for value in args.patch_split)
    if len(patch_split) != 2 or min(patch_split) <= 0:
        raise ValueError("patch-split requires two positive integers")
    if not torch.cuda.is_available():
        raise RuntimeError("ARAG prior generation requires CUDA")

    scene = _load_scene(scene_root, args.downscale)
    normal_folder = scene.image_folder.replace("images", "normals", 1)
    _ensure_publishable(scene_root, normal_folder, args.overwrite)
    staging_root = Path(tempfile.mkdtemp(
        prefix=".diffsoup-priors-", dir=scene_root,
    ))
    try:
        with _arag_inference_module() as infer_module:
            coarse_priors = _infer_coarse(infer_module, scene)
            report = _stage_refined_priors(
                infer_module, scene, coarse_priors, staging_root,
                normal_folder, checkpoint, patch_split,
            )
        _publish_scene_priors(
            scene_root, staging_root, normal_folder, args.overwrite,
        )
    finally:
        # Preserve a failed rollback in place; all other staging data is temporary.
        if not (staging_root / ".publish-backup").exists():
            shutil.rmtree(staging_root, ignore_errors=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ARAG geometry priors for a MipNeRF-360 scene.",
    )
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--checkpoint", default=str(_DEFAULT_CHECKPOINT))
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--patch-split", nargs=2, type=int, default=(2, 2))
    parser.add_argument("--overwrite", action="store_true")
    report = prepare_arag_scene(parser.parse_args())
    print(f"[prepared] frames={report['frames']}")
