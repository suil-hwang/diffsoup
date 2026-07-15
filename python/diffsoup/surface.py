# python/diffsoup/surface.py
"""Sparse differentiable surface expectations from fixed raster fragments."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch
import torch.nn.functional as F


class VertexExpectedSurfaceSamples(NamedTuple):
    """Sparse opacity-weighted depth, normal, and coverage render maps."""

    pixels_b_y_x: torch.Tensor
    expected_camera_z: torch.Tensor
    rendered_normal_camera: torch.Tensor
    accumulated_opacity: torch.Tensor
    valid: torch.Tensor


def _fragment_live_geometry(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    triangle_ids: torch.Tensor,
    pixels_b_y_x: torch.Tensor,
    K: torch.Tensor,
    Tcw: torch.Tensor,
    eps: float,
    parallel_cos_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute camera-Z and camera normal for fixed fragment identities."""
    # Clamp invalid IDs only for indexing; validity removes their contribution.
    triangle_valid = (triangle_ids >= 0) & (triangle_ids < faces.shape[0])
    safe_triangle_ids = triangle_ids.clamp(0, faces.shape[0] - 1)
    selected_faces = faces[safe_triangle_ids]
    p0 = vertices[selected_faces[:, 0]]
    p1 = vertices[selected_faces[:, 1]]
    p2 = vertices[selected_faces[:, 2]]

    # Live vertex positions define the differentiable plane and face normal.
    normal_raw = torch.linalg.cross(p1 - p0, p2 - p0, dim=-1)
    normal_length = torch.linalg.vector_norm(normal_raw, dim=-1)
    normal_world = normal_raw / normal_length.clamp_min(eps).unsqueeze(-1)

    # Cameras and pixels are fixed supervision geometry.
    with torch.no_grad():
        batch = pixels_b_y_x[:, 0]
        y = pixels_b_y_x[:, 1]
        x = pixels_b_y_x[:, 2]
        camera = Tcw.detach()[batch]
        intrinsics = K.detach()[batch]
        Rcw = camera[:, :3, :3]
        tcw = camera[:, :3, 3]
        camera_center_world = -torch.matmul(
            Rcw.transpose(1, 2), tcw.unsqueeze(-1),
        ).squeeze(-1)
        pixel_x = x.to(vertices.dtype) + 0.5
        pixel_y = y.to(vertices.dtype) + 0.5
        direction_camera = torch.stack(
            (
                (pixel_x - intrinsics[:, 0, 2]) / intrinsics[:, 0, 0],
                (pixel_y - intrinsics[:, 1, 2]) / intrinsics[:, 1, 1],
                torch.ones_like(pixel_x),
            ),
            dim=-1,
        )
        direction_world = torch.matmul(
            Rcw.transpose(1, 2), direction_camera.unsqueeze(-1),
        ).squeeze(-1)

    # Intersect each fixed pixel ray with its live triangle plane.
    denominator = (normal_world * direction_world).sum(dim=-1)
    direction_unit = F.normalize(direction_world, dim=-1, eps=eps)
    parallel_cosine = (normal_world * direction_unit).sum(dim=-1).abs()
    safe_denominator = torch.where(
        parallel_cosine > parallel_cos_eps,
        denominator,
        torch.ones_like(denominator),
    )
    camera_z = (
        normal_world * (p0 - camera_center_world)
    ).sum(dim=-1) / safe_denominator

    # Rotate normals to camera space and orient them toward the camera.
    normal_camera = torch.matmul(
        Rcw, normal_world.unsqueeze(-1),
    ).squeeze(-1)
    points_away = (normal_camera * direction_camera).sum(dim=-1) > 0
    normal_camera = torch.where(
        points_away.unsqueeze(-1), -normal_camera, normal_camera,
    )
    normal_camera = F.normalize(normal_camera, dim=-1, eps=eps)

    valid = (
        triangle_valid
        & (normal_length > eps)
        & (parallel_cosine > parallel_cos_eps)
        & (camera_z > eps)
        & torch.isfinite(camera_z)
        & torch.isfinite(normal_camera).all(dim=-1)
    )
    camera_z = torch.where(valid, camera_z, torch.zeros_like(camera_z))
    normal_camera = torch.where(
        valid.unsqueeze(-1), normal_camera, torch.zeros_like(normal_camera),
    )
    return camera_z, normal_camera, valid


def _blend_expected_fragments(
    groups: torch.Tensor,
    sort_depth: torch.Tensor,
    camera_z: torch.Tensor,
    normal_camera: torch.Tensor,
    alpha: torch.Tensor,
    geometry_valid: torch.Tensor,
    unique_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite exact fixed-opacity expectations for sampled pixels.

    Raster depth fixes visibility order. Invalid live geometry still consumes
    its raster opacity, but its undefined depth and normal do not contribute.
    The returned completeness mask rejects such partial expected maps.
    """
    # Group pixels while preserving the renderer's detached depth order.
    depth_order = torch.argsort(sort_depth.detach(), stable=True)
    group_order = torch.argsort(groups[depth_order], stable=True)
    order = depth_order[group_order]
    groups = groups[order]
    camera_z = camera_z[order]
    normal_camera = normal_camera[order]
    alpha = alpha[order].detach().clamp(0.0, 1.0)
    geometry_valid = geometry_valid[order]

    # Build an exclusive segmented log product without approximating alpha=1.
    opaque = alpha == 1.0
    safe_alpha = torch.where(opaque, torch.zeros_like(alpha), alpha)
    log_survival = torch.log1p(-safe_alpha)
    prefix = torch.cumsum(log_survival, dim=0, dtype=torch.float64)
    exclusive_global = prefix - log_survival.to(torch.float64)
    group_start = torch.ones_like(groups, dtype=torch.bool)
    group_start[1:] = groups[1:] != groups[:-1]
    segment = torch.cumsum(group_start.to(torch.int64), dim=0) - 1
    start_indices = torch.nonzero(group_start, as_tuple=False).squeeze(-1)
    exclusive_log = (
        exclusive_global - exclusive_global[start_indices][segment]
    ).clamp_max(0.0).to(alpha.dtype)

    # A preceding opaque fragment makes transmittance exactly zero.
    opaque_prefix = torch.cumsum(opaque.to(torch.int64), dim=0)
    opaque_exclusive = opaque_prefix - opaque.to(torch.int64)
    opaque_before = (
        opaque_exclusive - opaque_exclusive[start_indices][segment]
    ) > 0
    transmittance = torch.where(
        opaque_before,
        torch.zeros_like(alpha),
        torch.exp(exclusive_log),
    )
    weight = alpha * transmittance

    # Visibility includes every raster fragment; geometry values require validity.
    contribution_weight = weight * geometry_valid.to(weight.dtype)
    expected = torch.zeros(
        unique_count, dtype=camera_z.dtype, device=camera_z.device,
    ).index_add(0, groups, contribution_weight * camera_z)
    rendered_normal = torch.zeros(
        unique_count, 3, dtype=camera_z.dtype, device=camera_z.device,
    ).index_add(
        0, groups, contribution_weight.unsqueeze(-1) * normal_camera,
    )
    opacity = torch.zeros(
        unique_count, dtype=camera_z.dtype, device=camera_z.device,
    ).index_add(0, groups, weight)
    invalid_visibility = torch.zeros(
        unique_count, dtype=camera_z.dtype, device=camera_z.device,
    ).index_add(
        0, groups, weight * (~geometry_valid).to(weight.dtype),
    )
    completeness_eps = 8.0 * torch.finfo(camera_z.dtype).eps
    complete = invalid_visibility <= completeness_eps
    return expected, rendered_normal, opacity, complete


def vertex_expected_surface_samples(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor,
    frag_alpha: torch.Tensor,
    K: torch.Tensor,
    Tcw: torch.Tensor,
    pixels_b_y_x: torch.Tensor,
    image_size: Tuple[int, int],
    *,
    eps: float = 1e-8,
    parallel_cos_eps: float = 1e-3,
) -> VertexExpectedSurfaceSamples:
    """Evaluate sparse expected depth, normal, and opacity maps.

    Fragment membership, raster ordering, opacity, pixels, and cameras are
    detached. Only live ray-plane intersections and face normals differentiate.
    """
    assert vertices.ndim == 2 and vertices.shape[-1] == 3
    assert faces.ndim == 2 and faces.shape[-1] == 3
    assert frag_pix.ndim == 2 and frag_pix.shape[-1] == 3
    assert frag_attrs.shape == (frag_pix.shape[0], 4)
    assert frag_alpha.shape == (frag_pix.shape[0],)
    assert pixels_b_y_x.ndim == 2 and pixels_b_y_x.shape[-1] == 3
    assert faces.dtype in (torch.int32, torch.int64)
    assert frag_pix.dtype in (torch.int32, torch.int64)
    assert pixels_b_y_x.dtype in (torch.int32, torch.int64)
    height, width = image_size
    assert height > 0 and width > 0
    device = vertices.device
    tensors = (faces, frag_pix, frag_attrs, frag_alpha, K, Tcw, pixels_b_y_x)
    assert all(tensor.device == device for tensor in tensors)
    assert vertices.is_floating_point() and frag_attrs.is_floating_point()
    assert frag_alpha.dtype == vertices.dtype

    assert Tcw.ndim == 3 and Tcw.shape[1:] == (4, 4)
    assert Tcw.is_floating_point()
    batch_size = Tcw.shape[0]
    if K.ndim == 2:
        assert K.shape == (3, 3)
        K = K.unsqueeze(0).expand(batch_size, -1, -1)
    else:
        assert K.shape == (batch_size, 3, 3)
    assert K.dtype == Tcw.dtype == vertices.dtype

    # Sample identities stay fixed; reconstructed geometry remains live.
    pixels = pixels_b_y_x.detach().to(torch.int64)
    sample_count = pixels.shape[0]
    connected_zero = vertices.sum() * 0.0

    def empty_result() -> VertexExpectedSurfaceSamples:
        return VertexExpectedSurfaceSamples(
            pixels,
            torch.zeros(sample_count, dtype=vertices.dtype, device=device)
            + connected_zero,
            torch.zeros(sample_count, 3, dtype=vertices.dtype, device=device)
            + connected_zero,
            torch.zeros(sample_count, dtype=vertices.dtype, device=device),
            torch.zeros(sample_count, dtype=torch.bool, device=device),
        )

    if (
        sample_count == 0
        or vertices.shape[0] == 0
        or faces.shape[0] == 0
        or frag_pix.shape[0] == 0
    ):
        return empty_result()

    # Flatten (batch, y, x) into sortable keys for fragment lookup.
    sample_batch, sample_y, sample_x = pixels.unbind(dim=-1)
    sample_in_bounds = (
        (sample_batch >= 0)
        & (sample_batch < batch_size)
        & (sample_y >= 0)
        & (sample_y < height)
        & (sample_x >= 0)
        & (sample_x < width)
    )
    safe_batch = sample_batch.clamp(0, batch_size - 1)
    safe_y = sample_y.clamp(0, height - 1)
    safe_x = sample_x.clamp(0, width - 1)
    sample_keys = safe_batch * (height * width) + safe_y * width + safe_x
    unique_keys, sample_inverse = torch.unique(
        sample_keys, sorted=True, return_inverse=True,
    )

    # Match raster fragments whose pixel key appears in the sample set.
    detached_frag_pix = frag_pix.detach().to(torch.int64)
    frag_in_bounds = (
        (detached_frag_pix[:, 0] >= 0)
        & (detached_frag_pix[:, 0] < batch_size)
        & (detached_frag_pix[:, 1] >= 0)
        & (detached_frag_pix[:, 1] < height)
        & (detached_frag_pix[:, 2] >= 0)
        & (detached_frag_pix[:, 2] < width)
    )
    frag_keys = (
        detached_frag_pix[:, 0] * (height * width)
        + detached_frag_pix[:, 1] * width
        + detached_frag_pix[:, 2]
    )
    detached_raster_depth = frag_attrs[:, 2].detach()
    raster_depth_valid = (
        torch.isfinite(detached_raster_depth) & (detached_raster_depth > -1.0)
    )
    positions = torch.searchsorted(unique_keys, frag_keys)
    safe_positions = positions.clamp_max(unique_keys.shape[0] - 1)
    selected = (
        frag_in_bounds
        & raster_depth_valid
        & (positions < unique_keys.shape[0])
        & (unique_keys[safe_positions] == frag_keys)
    )
    fragment_indices = torch.nonzero(selected, as_tuple=False).squeeze(-1)
    if fragment_indices.numel() == 0:
        return empty_result()

    # Recompute live geometry while preserving raster visibility data.
    groups = positions[fragment_indices]
    fragment_pixels = detached_frag_pix[fragment_indices]
    selected_attrs = frag_attrs[fragment_indices].detach()
    triangle_ids = selected_attrs[:, 3].to(torch.int64) - 1
    sort_depth = selected_attrs[:, 2]
    camera_z, normal_camera, geometry_valid = _fragment_live_geometry(
        vertices, faces, triangle_ids, fragment_pixels, K, Tcw,
        eps, parallel_cos_eps,
    )
    expected_unique, normal_unique, opacity_unique, complete_unique = (
        _blend_expected_fragments(
            groups,
            sort_depth,
            camera_z,
            normal_camera,
            frag_alpha[fragment_indices],
            geometry_valid,
            unique_keys.shape[0],
        )
    )

    # Expand unique-pixel composites to the caller's sampled row order.
    expected = expected_unique[sample_inverse]
    rendered_normal = normal_unique[sample_inverse]
    opacity = opacity_unique[sample_inverse]
    complete = complete_unique[sample_inverse]
    valid = (
        sample_in_bounds
        & complete
        & (opacity > eps)
        & (expected > eps)
        & torch.isfinite(expected)
        & torch.isfinite(rendered_normal).all(dim=-1)
    )
    expected = torch.where(valid, expected, torch.zeros_like(expected))
    rendered_normal = torch.where(
        valid.unsqueeze(-1), rendered_normal, torch.zeros_like(rendered_normal),
    )
    opacity = torch.where(sample_in_bounds, opacity, torch.zeros_like(opacity))
    return VertexExpectedSurfaceSamples(
        pixels, expected, rendered_normal, opacity, valid,
    )
