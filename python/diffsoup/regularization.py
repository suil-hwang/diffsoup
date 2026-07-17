# python/diffsoup/regularization.py
"""Losses for sparse geometry-prior supervision."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .surface import VertexExpectedSurfaceSamples


# ---------------------------------------------------------------------------
#  Normal-prior loss
# ---------------------------------------------------------------------------

def normal_prior_loss(
    surface: VertexExpectedSurfaceSamples,
    prior_normal_camera: torch.Tensor,
    prior_valid: torch.Tensor,
) -> torch.Tensor:
    """Return the expected per-fragment oriented normal-prior loss.

    ``opacity - cosine = Σ_i w_i (1 - n_i · n_prior)`` with detached
    visibility weights ``w_i``; only the live geometry receives gradients.

    Args:
        surface:     Expected surface samples from
                     :func:`diffsoup.vertex_expected_surface_samples`.
        prior_normal_camera: Unit camera-space prior normals ``(N, 3)``,
                     detached.
        prior_valid: Boolean mask ``(N,)`` of usable prior rows.

    Returns:
        Scalar mean over all ``N`` sampled rows; invalid rows count as zero.
    """
    prior_normal_camera = prior_normal_camera.detach()
    assert prior_normal_camera.shape == surface.rendered_normal_camera.shape
    assert prior_normal_camera.device == surface.rendered_normal_camera.device
    assert prior_valid.shape == surface.valid.shape
    prior_valid = prior_valid.detach().to(
        device=surface.valid.device, dtype=torch.bool,
    )
    valid = prior_valid & surface.valid
    safe_prior = torch.where(
        valid.unsqueeze(-1),
        prior_normal_camera,
        torch.zeros_like(prior_normal_camera),
    )
    rendered = torch.where(
        valid.unsqueeze(-1),
        surface.rendered_normal_camera,
        torch.zeros_like(surface.rendered_normal_camera),
    )
    cosine = (rendered * safe_prior).sum(dim=-1)
    opacity = surface.accumulated_opacity.detach()
    penalty = torch.where(
        valid, opacity - cosine, torch.zeros_like(cosine),
    )
    if penalty.numel() == 0:
        return surface.expected_camera_z.sum() * 0.0
    return penalty.mean()


# ---------------------------------------------------------------------------
#  Inverse-depth-prior loss
# ---------------------------------------------------------------------------

def inverse_depth_prior_loss(
    surface: VertexExpectedSurfaceSamples,
    prior_inverse_depth: torch.Tensor,
    prior_valid: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return masked L1 loss on inverse conditional expected camera depth.

    ``|1 / (z + eps) - prior|`` with ``z = expected_camera_z / opacity``
    (detached, floored) — opacity-invariant supervision in inverse-depth
    space; only the live geometry receives gradients.

    Args:
        surface:     Expected surface samples from
                     :func:`diffsoup.vertex_expected_surface_samples`.
        prior_inverse_depth: Prior inverse depths ``(N,)``, detached.
        prior_valid: Boolean mask ``(N,)`` of usable prior rows.
        eps:         Reciprocal stabiliser and opacity-floor lower bound.

    Returns:
        Scalar mean over all ``N`` sampled rows; invalid or low-opacity
        rows count as zero.
    """
    assert eps >= 0
    prior_inverse_depth = prior_inverse_depth.detach()
    assert prior_inverse_depth.shape == surface.expected_camera_z.shape
    assert prior_inverse_depth.device == surface.expected_camera_z.device
    assert prior_valid.shape == surface.valid.shape
    prior_valid = prior_valid.detach().to(
        device=surface.valid.device, dtype=torch.bool,
    )
    safe_prior = torch.where(
        prior_valid, prior_inverse_depth, torch.zeros_like(prior_inverse_depth),
    )
    # Use D / A for opacity-invariant depth while keeping alpha detached.
    opacity = surface.accumulated_opacity.detach()
    opacity_floor = max(eps, torch.finfo(opacity.dtype).tiny)
    valid = (
        prior_valid
        & surface.valid
        & torch.isfinite(opacity)
        & (opacity > opacity_floor)
    )
    conditional_camera_z = (
        surface.expected_camera_z / opacity.clamp_min(opacity_floor)
    )
    safe_camera_z = torch.where(
        valid, conditional_camera_z, torch.ones_like(conditional_camera_z),
    )
    predicted = (safe_camera_z + eps).reciprocal()
    penalty = (predicted - safe_prior).abs()
    # Ignore missing surfaces because they have no vertex gradient.
    penalty = torch.where(valid, penalty, torch.zeros_like(penalty))
    if penalty.numel() == 0:
        return surface.expected_camera_z.sum() * 0.0
    return penalty.mean()
