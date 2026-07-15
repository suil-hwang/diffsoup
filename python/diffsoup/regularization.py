# python/diffsoup/regularization.py
"""Losses for sparse geometry-prior supervision."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .surface import VertexExpectedSurfaceSamples


def normal_prior_loss(
    surface: VertexExpectedSurfaceSamples,
    prior_normal_camera: torch.Tensor,
    prior_valid: torch.Tensor,
) -> torch.Tensor:
    """Return the oriented opacity-weighted normal-prior loss."""
    prior_normal_camera = prior_normal_camera.detach()
    assert prior_normal_camera.shape == surface.rendered_normal_camera.shape
    assert prior_normal_camera.device == surface.rendered_normal_camera.device
    assert prior_valid.shape == surface.valid.shape
    prior_valid = prior_valid.detach().to(
        device=surface.valid.device, dtype=torch.bool,
    )
    safe_prior = torch.where(
        prior_valid.unsqueeze(-1),
        prior_normal_camera,
        torch.zeros_like(prior_normal_camera),
    )
    rendered = torch.where(
        surface.valid.unsqueeze(-1),
        surface.rendered_normal_camera,
        torch.zeros_like(surface.rendered_normal_camera),
    )
    cosine = (rendered * safe_prior).sum(dim=-1)
    penalty = torch.where(
        prior_valid, 1.0 - cosine, torch.zeros_like(cosine),
    )
    if penalty.numel() == 0:
        return surface.expected_camera_z.sum() * 0.0
    return penalty.mean()


def inverse_depth_prior_loss(
    surface: VertexExpectedSurfaceSamples,
    prior_inverse_depth: torch.Tensor,
    prior_valid: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return masked L1 loss on inverse unnormalized expected camera depth."""
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
    predicted = (surface.expected_camera_z + eps).reciprocal()
    penalty = (predicted - safe_prior).abs()
    # Missing expected surfaces have no vertex gradient under the fixed-opacity
    # contract, so exclude their otherwise enormous constant reciprocal value.
    valid = prior_valid & surface.valid
    penalty = torch.where(valid, penalty, torch.zeros_like(penalty))
    if penalty.numel() == 0:
        return surface.expected_camera_z.sum() * 0.0
    return penalty.mean()
