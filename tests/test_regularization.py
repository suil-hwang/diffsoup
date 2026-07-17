from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

import diffsoup as ds


_SURFACE_SPEC = importlib.util.spec_from_file_location(
    "diffsoup_surface_under_test",
    Path(__file__).parents[1] / "python" / "diffsoup" / "surface.py",
)
assert _SURFACE_SPEC is not None and _SURFACE_SPEC.loader is not None
surface = importlib.util.module_from_spec(_SURFACE_SPEC)
sys.modules[_SURFACE_SPEC.name] = surface
_SURFACE_SPEC.loader.exec_module(surface)

_SPEC = importlib.util.spec_from_file_location(
    "diffsoup_regularization_under_test",
    Path(__file__).parents[1] / "python" / "diffsoup" / "regularization.py",
)
assert _SPEC is not None and _SPEC.loader is not None
reg = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = reg
_SPEC.loader.exec_module(reg)

_UTILS_SPEC = importlib.util.spec_from_file_location(
    "diffsoup_examples_utils_under_test",
    Path(__file__).parents[1] / "examples" / "utils.py",
)
assert _UTILS_SPEC is not None and _UTILS_SPEC.loader is not None
example_utils = importlib.util.module_from_spec(_UTILS_SPEC)
sys.modules[_UTILS_SPEC.name] = example_utils
_UTILS_SPEC.loader.exec_module(example_utils)


def test_projection_reproduces_noncentered_pixel_coordinates():
    height, width = 60, 100
    K = torch.tensor(
        [[80.0, 0.0, 43.0], [0.0, 70.0, 19.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    Tcw = torch.eye(4, dtype=torch.float64)
    u, v, z = 30.5, 42.5, 4.0
    point_camera = torch.tensor(
        [(u - K[0, 2]) * z / K[0, 0],
         (v - K[1, 2]) * z / K[1, 1],
         z, 1.0],
        dtype=torch.float64,
    )
    mvp = example_utils.mvp_from_K_Tcw(
        K, Tcw, (height, width), z_near=0.1, z_far=100.0, flip_z=True,
    )
    clip = mvp @ point_camera
    ndc = clip[:3] / clip[3]
    recovered_u = 0.5 * (ndc[0] + 1.0) * width
    recovered_v = 0.5 * (ndc[1] + 1.0) * height
    torch.testing.assert_close(recovered_u, torch.tensor(u, dtype=K.dtype))
    torch.testing.assert_close(recovered_v, torch.tensor(v, dtype=K.dtype))


def test_expected_surface_face_forward_normal_is_winding_invariant():
    dtype = torch.float64
    vertices = torch.tensor(
        [[-2.0, -2.0, 3.0], [2.0, -2.0, 3.0], [0.0, 2.0, 3.0]],
        dtype=dtype,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    frag_pix = torch.tensor([[0, 0, 0]], dtype=torch.int32)
    frag_attrs = torch.tensor([[0.25, 0.25, 0.0, 1.0]], dtype=dtype)
    frag_alpha = torch.tensor([0.5], dtype=dtype)
    K = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0)
    pixels = torch.tensor([[0, 0, 0]], dtype=torch.int64)

    forward = surface.vertex_expected_surface_samples(
        vertices, faces, frag_pix, frag_attrs, frag_alpha,
        K, Tcw, pixels, (1, 1),
    )
    reverse = surface.vertex_expected_surface_samples(
        vertices, faces[:, [0, 2, 1]], frag_pix, frag_attrs, frag_alpha,
        K, Tcw, pixels, (1, 1),
    )

    assert forward.valid.item() and reverse.valid.item()
    torch.testing.assert_close(forward.expected_camera_z, reverse.expected_camera_z)
    torch.testing.assert_close(
        forward.rendered_normal_camera, reverse.rendered_normal_camera,
    )
    torch.testing.assert_close(
        forward.rendered_normal_camera,
        torch.tensor([[0.0, 0.0, -0.5]], dtype=dtype),
    )


def test_expected_surface_matches_closed_form_alpha_compositing():
    dtype = torch.float64
    vertices = torch.tensor(
        [
            [-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0],
            [-1.0, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    frag_pix = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.int32)
    frag_attrs = torch.tensor(
        [[0.25, 0.25, 0.0, 1.0], [0.25, 0.25, 0.0, 2.0]],
        dtype=dtype,
    )
    frag_alpha = torch.tensor(
        [0.25, 0.5], dtype=dtype, requires_grad=True,
    )
    K = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=dtype, requires_grad=True,
    )
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0).requires_grad_(True)
    pixels = torch.tensor([[0, 0, 0]], dtype=torch.int64)

    expected_surface = surface.vertex_expected_surface_samples(
        vertices, faces, frag_pix, frag_attrs, frag_alpha,
        K, Tcw, pixels, (1, 1),
    )
    assert expected_surface.valid.item()
    torch.testing.assert_close(
        expected_surface.expected_camera_z, torch.tensor([2.0], dtype=dtype),
    )
    torch.testing.assert_close(
        expected_surface.accumulated_opacity,
        torch.tensor([0.625], dtype=dtype),
    )
    torch.testing.assert_close(
        expected_surface.rendered_normal_camera,
        torch.tensor([[0.0, 0.0, -0.625]], dtype=dtype),
    )
    normal_prior = torch.tensor(
        [[0.0, 0.0, -1.0]], dtype=dtype, requires_grad=True,
    )
    depth_prior = torch.tensor([0.4], dtype=dtype, requires_grad=True)
    normal_loss = reg.normal_prior_loss(
        expected_surface,
        normal_prior,
        torch.ones_like(expected_surface.valid),
    )
    depth_loss = reg.inverse_depth_prior_loss(
        expected_surface,
        depth_prior,
        torch.ones_like(expected_surface.valid),
    )
    torch.testing.assert_close(
        normal_loss, torch.tensor(0.0, dtype=dtype),
    )
    conditional_camera_z = 2.0 / 0.625
    torch.testing.assert_close(
        depth_loss,
        torch.tensor(
            abs(1.0 / (conditional_camera_z + 1e-6) - 0.4),
            dtype=dtype,
        ),
    )
    per_fragment_depth_loss = (
        0.25 * abs(0.5 - 0.4) + 0.375 * abs(0.25 - 0.4)
    )
    assert float(depth_loss.detach()) != pytest.approx(per_fragment_depth_loss)

    (normal_loss + depth_loss).backward()
    assert vertices.grad is not None
    assert torch.isfinite(vertices.grad).all()
    assert vertices.grad.abs().sum() > 0
    assert frag_alpha.grad is None
    assert K.grad is None
    assert Tcw.grad is None
    assert normal_prior.grad is None
    assert depth_prior.grad is None


def test_normal_loss_masks_invalid_surfaces_and_keeps_sample_count_denominator():
    dtype = torch.float64
    rendered_normal = torch.tensor(
        [[0.0, 0.0, -0.5], [0.0, -0.9, 0.0], [0.4, 0.0, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    expected_surface = surface.VertexExpectedSurfaceSamples(
        torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
        torch.ones(3, dtype=dtype, requires_grad=True),
        rendered_normal,
        torch.tensor([0.75, 0.9, 0.4], dtype=dtype),
        torch.tensor([True, False, True]),
    )
    prior = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    prior_valid = torch.tensor([True, True, False])

    loss = reg.normal_prior_loss(expected_surface, prior, prior_valid)

    # Only the valid surface contributes 0.75 - 0.5. Missing surfaces and
    # invalid priors contribute zero while the sampled-row denominator remains.
    torch.testing.assert_close(loss, torch.tensor(1.0 / 12.0, dtype=dtype))
    loss.backward()
    torch.testing.assert_close(
        rendered_normal.grad,
        torch.tensor(
            [[0.0, 0.0, 1.0 / 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=dtype,
        ),
    )
    assert prior.grad is None


@pytest.mark.parametrize("opacity_value", [0.25, 0.5, 0.8, 1.0])
def test_normal_loss_has_no_opacity_floor_and_detaches_opacity(
    opacity_value: float,
):
    dtype = torch.float64
    theta = torch.tensor(0.3, dtype=dtype, requires_grad=True)
    opacity = torch.tensor(
        [opacity_value], dtype=dtype, requires_grad=True,
    )
    direction = torch.stack((
        theta.sin(),
        torch.zeros_like(theta),
        -theta.cos(),
    )).unsqueeze(0)
    expected_surface = surface.VertexExpectedSurfaceSamples(
        torch.tensor([[0, 0, 0]]),
        torch.ones(1, dtype=dtype),
        opacity.detach().unsqueeze(-1) * direction,
        opacity,
        torch.ones(1, dtype=torch.bool),
    )

    loss = reg.normal_prior_loss(
        expected_surface,
        torch.tensor([[0.0, 0.0, -1.0]], dtype=dtype),
        torch.ones(1, dtype=torch.bool),
    )

    torch.testing.assert_close(
        loss,
        opacity.detach().squeeze(0) * (1.0 - theta.detach().cos()),
    )
    loss.backward()
    torch.testing.assert_close(
        theta.grad,
        opacity.detach().squeeze(0) * theta.detach().sin(),
    )
    assert opacity.grad is None


@pytest.mark.parametrize("opacity_value", [0.3, 0.5, 0.8])
def test_conditional_inverse_depth_is_opacity_invariant_and_detaches_opacity(
    opacity_value: float,
):
    dtype = torch.float64
    camera_z = torch.tensor([1.5], dtype=dtype, requires_grad=True)
    opacity = torch.tensor(
        [opacity_value], dtype=dtype, requires_grad=True,
    )
    expected_surface = surface.VertexExpectedSurfaceSamples(
        torch.tensor([[0, 0, 0]]),
        opacity.detach() * camera_z,
        torch.zeros(1, 3, dtype=dtype),
        opacity,
        torch.ones(1, dtype=torch.bool),
    )

    loss = reg.inverse_depth_prior_loss(
        expected_surface,
        torch.tensor([0.5], dtype=dtype),
        torch.ones(1, dtype=torch.bool),
        eps=0.0,
    )

    torch.testing.assert_close(
        loss, torch.tensor(abs(1.0 / 1.5 - 0.5), dtype=dtype),
    )
    loss.backward()
    assert camera_z.grad is not None and camera_z.grad.item() < 0.0
    assert opacity.grad is None


def test_expected_surface_background_depth_loss_is_zero_and_connected():
    dtype = torch.float64
    vertices = torch.empty((0, 3), dtype=dtype, requires_grad=True)
    faces = torch.empty((0, 3), dtype=torch.int64)
    frag_pix = torch.empty((0, 3), dtype=torch.int32)
    frag_attrs = torch.empty((0, 4), dtype=dtype)
    frag_alpha = torch.empty((0,), dtype=dtype)
    K = torch.eye(3, dtype=dtype)
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0)
    pixels = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    expected_surface = surface.vertex_expected_surface_samples(
        vertices, faces, frag_pix, frag_attrs, frag_alpha,
        K, Tcw, pixels, (1, 1),
    )
    depth_loss = reg.inverse_depth_prior_loss(
        expected_surface,
        torch.tensor([0.5], dtype=dtype),
        torch.ones_like(expected_surface.valid),
    )
    torch.testing.assert_close(depth_loss, torch.tensor(0.0, dtype=dtype))
    depth_loss.backward()
    assert vertices.grad is not None


def test_saturated_float32_alpha_is_finite_across_pixel_groups():
    vertices = torch.tensor(
        [
            [-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0],
            [-1.0, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    frag_pix = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 1]],
        dtype=torch.int32,
    )
    frag_attrs = torch.tensor(
        [
            [0.25, 0.25, 0.0, 1.0], [0.25, 0.25, 0.0, 2.0],
            [0.25, 0.25, 0.0, 1.0], [0.25, 0.25, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    frag_alpha = torch.tensor(
        [1.0, 0.5, 1.0, 0.5], dtype=torch.float32, requires_grad=True,
    )
    K = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
    )
    Tcw = torch.eye(4).unsqueeze(0)
    pixels = torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int64)

    expected_surface = surface.vertex_expected_surface_samples(
        vertices, faces, frag_pix, frag_attrs, frag_alpha,
        K, Tcw, pixels, (1, 2),
    )
    assert expected_surface.valid.all()
    assert torch.isfinite(expected_surface.expected_camera_z).all()
    assert torch.isfinite(expected_surface.rendered_normal_camera).all()
    assert torch.isfinite(expected_surface.accumulated_opacity).all()
    torch.testing.assert_close(
        expected_surface.expected_camera_z, torch.full((2,), 2.0),
        rtol=1e-6, atol=1e-6,
    )
    torch.testing.assert_close(
        expected_surface.accumulated_opacity, torch.ones(2),
        rtol=1e-6, atol=1e-6,
    )

    expected_surface.expected_camera_z.sum().backward()
    assert vertices.grad is not None and torch.isfinite(vertices.grad).all()
    assert frag_alpha.grad is None


def test_exact_unit_alpha_completely_occludes_later_fragments():
    dtype = torch.float64
    groups = torch.tensor([0, 0], dtype=torch.int64)
    sort_depth = torch.tensor([1.0, 2.0], dtype=dtype)
    camera_z = torch.tensor([2.0, 4.0], dtype=dtype, requires_grad=True)
    normal = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=dtype,
        requires_grad=True,
    )
    alpha = torch.tensor([1.0, 0.5], dtype=dtype, requires_grad=True)
    geometry_valid = torch.ones(2, dtype=torch.bool)

    expected, rendered_normal, opacity, complete = (
        surface._blend_expected_fragments(
            groups,
            sort_depth,
            camera_z,
            normal,
            alpha,
            geometry_valid,
            unique_count=1,
        )
    )

    torch.testing.assert_close(expected, torch.tensor([2.0], dtype=dtype))
    torch.testing.assert_close(
        rendered_normal, torch.tensor([[0.0, 0.0, -1.0]], dtype=dtype),
    )
    torch.testing.assert_close(opacity, torch.ones(1, dtype=dtype))
    assert complete.item()

    (expected.sum() + rendered_normal.sum()).backward()
    torch.testing.assert_close(
        camera_z.grad, torch.tensor([1.0, 0.0], dtype=dtype),
    )
    assert normal.grad is not None and torch.isfinite(normal.grad).all()
    assert alpha.grad is None


def test_near_unit_alpha_preserves_residual_transmittance():
    dtype = torch.float32
    groups = torch.tensor([0, 0], dtype=torch.int64)
    sort_depth = torch.tensor([1.0, 2.0], dtype=dtype)
    camera_z = torch.tensor([0.0, 1.0], dtype=dtype, requires_grad=True)
    normal = torch.zeros(2, 3, dtype=dtype)
    almost_one = torch.nextafter(
        torch.tensor(1.0, dtype=dtype), torch.tensor(0.0, dtype=dtype),
    )
    alpha = torch.stack((almost_one, torch.tensor(0.5))).requires_grad_(True)
    geometry_valid = torch.ones(2, dtype=torch.bool)

    expected, _, opacity, complete = surface._blend_expected_fragments(
        groups,
        sort_depth,
        camera_z,
        normal,
        alpha,
        geometry_valid,
        unique_count=1,
    )
    back_weight = (1.0 - almost_one) * 0.5
    torch.testing.assert_close(expected, back_weight.unsqueeze(0))
    torch.testing.assert_close(
        opacity, (almost_one + back_weight).unsqueeze(0),
    )
    assert expected.item() > 0 and complete.item()

    expected.sum().backward()
    torch.testing.assert_close(
        camera_z.grad, torch.stack((almost_one, back_weight)),
    )
    assert alpha.grad is None


def test_fragment_compositor_uses_raster_depth_instead_of_live_camera_z():
    dtype = torch.float64
    groups = torch.tensor([0, 0], dtype=torch.int64)
    # Raster order says the z=4 fragment is in front even though live camera-Z
    # would sort the z=2 fragment first.
    sort_depth = torch.tensor(
        [2.0, 1.0], dtype=dtype, requires_grad=True,
    )
    camera_z = torch.tensor([2.0, 4.0], dtype=dtype, requires_grad=True)
    normal = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype,
    )
    alpha = torch.tensor([0.5, 0.5], dtype=dtype)
    geometry_valid = torch.ones(2, dtype=torch.bool)

    expected, rendered_normal, opacity, complete = (
        surface._blend_expected_fragments(
            groups,
            sort_depth,
            camera_z,
            normal,
            alpha,
            geometry_valid,
            unique_count=1,
        )
    )

    # Raster order [z=4, z=2] gives 0.5*4 + 0.25*2 = 2.5.
    torch.testing.assert_close(expected, torch.tensor([2.5], dtype=dtype))
    torch.testing.assert_close(
        rendered_normal, torch.tensor([[0.25, 0.5, 0.0]], dtype=dtype),
    )
    torch.testing.assert_close(opacity, torch.tensor([0.75], dtype=dtype))
    assert complete.item()
    expected.sum().backward()
    torch.testing.assert_close(
        camera_z.grad, torch.tensor([0.25, 0.5], dtype=dtype),
    )
    assert sort_depth.grad is None


def test_invalid_front_fragment_consumes_visibility_without_contributing():
    dtype = torch.float64
    groups = torch.tensor([0, 0], dtype=torch.int64)
    sort_depth = torch.tensor([1.0, 2.0], dtype=dtype)
    camera_z = torch.tensor([0.0, 4.0], dtype=dtype)
    normal = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=dtype,
    )
    alpha = torch.tensor([0.8, 0.5], dtype=dtype)
    geometry_valid = torch.tensor([False, True])

    expected, rendered_normal, opacity, complete = (
        surface._blend_expected_fragments(
            groups,
            sort_depth,
            camera_z,
            normal,
            alpha,
            geometry_valid,
            unique_count=1,
        )
    )
    # The invalid front fragment contributes no geometry but leaves only
    # (1 - 0.8) * 0.5 = 0.1 weight for the valid back fragment.
    torch.testing.assert_close(expected, torch.tensor([0.4], dtype=dtype))
    torch.testing.assert_close(
        rendered_normal, torch.tensor([[0.0, 0.0, -0.1]], dtype=dtype),
    )
    torch.testing.assert_close(opacity, torch.tensor([0.9], dtype=dtype))
    assert not complete.item()

    vertices = torch.tensor(
        [
            [-1.0, -1.0, 2.0], [0.0, 0.0, 2.0], [1.0, 1.0, 2.0],
            [-1.0, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)
    frag_pix = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.int32)
    frag_attrs = torch.tensor(
        [[0.25, 0.25, 1.0, 1.0], [0.25, 0.25, 2.0, 2.0]],
        dtype=dtype,
    )
    K = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0)
    pixels = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    expected_surface = surface.vertex_expected_surface_samples(
        vertices,
        faces,
        frag_pix,
        frag_attrs,
        alpha,
        K,
        Tcw,
        pixels,
        (1, 1),
    )
    assert not expected_surface.valid.item()
    torch.testing.assert_close(
        expected_surface.expected_camera_z, torch.zeros(1, dtype=dtype),
    )
    torch.testing.assert_close(
        expected_surface.rendered_normal_camera,
        torch.zeros(1, 3, dtype=dtype),
    )
    torch.testing.assert_close(
        expected_surface.accumulated_opacity, torch.tensor([0.9], dtype=dtype),
    )


def test_expected_surface_excludes_native_invalid_raster_depths():
    dtype = torch.float64
    vertices = torch.tensor(
        [
            [-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0],
            [-1.0, -1.0, 4.0], [1.0, -1.0, 4.0], [0.0, 1.0, 4.0],
            [-1.0, -1.0, 6.0], [1.0, -1.0, 6.0], [0.0, 1.0, 6.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.int64,
    )
    frag_pix = torch.tensor([[0, 0, 0]] * 3, dtype=torch.int32)
    frag_attrs = torch.tensor(
        [
            [0.25, 0.25, -1.0, 1.0],
            [0.25, 0.25, float("nan"), 2.0],
            [0.25, 0.25, 0.25, 3.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    frag_alpha = torch.tensor([1.0, 1.0, 0.5], dtype=dtype)
    K = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0)
    pixels = torch.tensor([[0, 0, 0]], dtype=torch.int64)

    expected_surface = surface.vertex_expected_surface_samples(
        vertices,
        faces,
        frag_pix,
        frag_attrs,
        frag_alpha,
        K,
        Tcw,
        pixels,
        (1, 1),
    )

    assert expected_surface.valid.item()
    torch.testing.assert_close(
        expected_surface.expected_camera_z, torch.tensor([3.0], dtype=dtype),
    )
    torch.testing.assert_close(
        expected_surface.rendered_normal_camera,
        torch.tensor([[0.0, 0.0, -0.5]], dtype=dtype),
    )
    torch.testing.assert_close(
        expected_surface.accumulated_opacity, torch.tensor([0.5], dtype=dtype),
    )
    expected_surface.expected_camera_z.sum().backward()
    assert frag_attrs.grad is None


def test_fragment_compositor_matches_naive_front_to_back_reference():
    dtype = torch.float64
    groups = torch.tensor([0, 0, 1, 1, 0], dtype=torch.int64)
    camera_z = torch.tensor(
        [4.0, 2.0, 5.0, 1.0, 3.0], dtype=dtype, requires_grad=True,
    )
    normal = torch.tensor(
        [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    alpha = torch.tensor(
        [0.2, 0.3, 0.4, 0.5, 1.0], dtype=dtype, requires_grad=True,
    )
    sort_depth = camera_z.detach().clone()
    geometry_valid = torch.ones(camera_z.shape[0], dtype=torch.bool)
    expected, rendered_normal, opacity, complete = surface._blend_expected_fragments(
        groups, sort_depth, camera_z, normal, alpha, geometry_valid,
        unique_count=2,
    )

    reference_depth = torch.zeros(2, dtype=dtype)
    reference_normal = torch.zeros(2, 3, dtype=dtype)
    reference_opacity = torch.zeros(2, dtype=dtype)
    for group in range(2):
        indices = torch.nonzero(groups == group, as_tuple=False).squeeze(-1)
        indices = indices[torch.argsort(sort_depth[indices], stable=True)]
        transmittance = 1.0
        for index in indices.tolist():
            opacity_i = min(max(float(alpha[index].detach()), 0.0), 1.0)
            weight = transmittance * opacity_i
            reference_depth[group] += weight * camera_z.detach()[index]
            reference_normal[group] += weight * normal.detach()[index]
            reference_opacity[group] += weight
            transmittance *= 1.0 - opacity_i

    torch.testing.assert_close(expected.detach(), reference_depth)
    torch.testing.assert_close(rendered_normal.detach(), reference_normal)
    torch.testing.assert_close(opacity.detach(), reference_opacity)
    assert complete.all()
    (expected.sum() + rendered_normal.sum()).backward()
    assert camera_z.grad is not None and torch.isfinite(camera_z.grad).all()
    assert normal.grad is not None and torch.isfinite(normal.grad).all()
    assert alpha.grad is None


def test_fragment_compositor_is_stable_across_many_pixel_groups():
    group_count = 70_000
    groups = torch.arange(group_count).repeat_interleave(2)
    camera_z = torch.ones(group_count * 2)
    normal = torch.zeros(group_count * 2, 3)
    alpha = torch.full((group_count * 2,), 0.5)
    expected, _, opacity, complete = surface._blend_expected_fragments(
        groups,
        camera_z,
        camera_z,
        normal,
        alpha,
        torch.ones_like(groups, dtype=torch.bool),
        unique_count=group_count,
    )
    reference = torch.full((group_count,), 0.75)
    torch.testing.assert_close(expected, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(opacity, reference, rtol=1e-6, atol=1e-6)
    assert complete.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fragment_compositor_uses_nondefault_cuda_stream():
    device = torch.device("cuda")
    group_count = 70_000
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        groups = torch.arange(group_count, device=device).repeat_interleave(2)
        camera_z = torch.ones(
            group_count * 2, device=device, requires_grad=True,
        )
        normal = torch.zeros(group_count * 2, 3, device=device)
        alpha = torch.full(
            (group_count * 2,), 0.5, device=device, requires_grad=True,
        )
        expected, _, opacity, coverage_complete = surface._blend_expected_fragments(
            groups,
            camera_z,
            camera_z,
            normal,
            alpha,
            torch.ones_like(groups, dtype=torch.bool),
            unique_count=group_count,
        )
        expected.sum().backward()
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()

    reference = torch.full_like(expected, 0.75)
    torch.testing.assert_close(expected, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(opacity, reference, rtol=1e-6, atol=1e-6)
    assert coverage_complete.all()
    assert camera_z.grad is not None and torch.isfinite(camera_z.grad).all()
    assert alpha.grad is None


def test_expected_surface_uses_camera_z_and_camera_space_normal():
    dtype = torch.float64
    angle = torch.tensor(0.4, dtype=dtype)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    Rcw = torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=dtype,
    )
    translation = torch.tensor([0.3, -0.2, 0.5], dtype=dtype)
    camera_vertices = torch.tensor(
        [[-1.0, -1.0, 3.0], [1.0, -1.0, 3.0], [0.0, 1.0, 3.0]],
        dtype=dtype,
    )
    vertices = (
        (camera_vertices - translation) @ Rcw
    ).requires_grad_(True)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    frag_pix = torch.tensor([[0, 42, 30]], dtype=torch.int32)
    frag_attrs = torch.tensor([[0.2, 0.3, 0.0, 1.0]], dtype=dtype)
    frag_alpha = torch.tensor([0.5], dtype=dtype)
    K = torch.tensor(
        [[80.0, 0.0, 43.0], [0.0, 70.0, 19.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0)
    Tcw[0, :3, :3] = Rcw
    Tcw[0, :3, 3] = translation
    pixels = torch.tensor([[0, 42, 30]], dtype=torch.int64)

    expected_surface = surface.vertex_expected_surface_samples(
        vertices, faces, frag_pix, frag_attrs, frag_alpha,
        K, Tcw, pixels, (60, 100),
    )
    assert expected_surface.valid.item()
    torch.testing.assert_close(
        expected_surface.expected_camera_z, torch.tensor([1.5], dtype=dtype),
    )
    torch.testing.assert_close(
        expected_surface.rendered_normal_camera,
        torch.tensor([[0.0, 0.0, -0.5]], dtype=dtype),
        rtol=1e-12, atol=1e-12,
    )


def test_expected_inverse_depth_vertex_gradient_matches_finite_difference():
    dtype = torch.float64
    base_vertices = torch.tensor(
        [[-2.0, -2.0, 2.5], [2.0, -2.0, 3.5], [0.0, 2.0, 3.0]],
        dtype=dtype,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    frag_pix = torch.tensor([[0, 0, 0]], dtype=torch.int32)
    frag_attrs = torch.tensor([[0.2, 0.3, 0.0, 1.0]], dtype=dtype)
    frag_alpha = torch.tensor([0.6], dtype=dtype)
    K = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0)
    pixels = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    prior = torch.tensor([0.4], dtype=dtype)

    def evaluate(vertices: torch.Tensor) -> torch.Tensor:
        expected_surface = surface.vertex_expected_surface_samples(
            vertices, faces, frag_pix, frag_attrs, frag_alpha,
            K, Tcw, pixels, (2, 2),
        )
        return reg.inverse_depth_prior_loss(
            expected_surface, prior, torch.ones_like(expected_surface.valid),
        )

    vertices = base_vertices.clone().requires_grad_(True)
    analytic = torch.autograd.grad(evaluate(vertices), vertices)[0][0, 2]
    delta = 1e-5
    plus = base_vertices.clone()
    minus = base_vertices.clone()
    plus[0, 2] += delta
    minus[0, 2] -= delta
    numeric = (evaluate(plus) - evaluate(minus)) / (2.0 * delta)
    torch.testing.assert_close(analytic, numeric, rtol=2e-4, atol=2e-6)


def test_expected_normal_vertex_gradient_matches_finite_difference():
    dtype = torch.float64
    base_vertices = torch.tensor(
        [[-2.0, -2.0, 2.5], [2.0, -2.0, 3.5], [0.0, 2.0, 3.0]],
        dtype=dtype,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    frag_pix = torch.tensor([[0, 0, 0]], dtype=torch.int32)
    frag_attrs = torch.tensor([[0.2, 0.3, 0.0, 1.0]], dtype=dtype)
    frag_alpha = torch.tensor([0.6], dtype=dtype)
    K = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0)
    pixels = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    prior = torch.tensor([[0.3, 0.4, -0.8]], dtype=dtype)
    prior = prior / torch.linalg.vector_norm(prior, dim=-1, keepdim=True)

    def evaluate(vertices: torch.Tensor) -> torch.Tensor:
        expected_surface = surface.vertex_expected_surface_samples(
            vertices, faces, frag_pix, frag_attrs, frag_alpha,
            K, Tcw, pixels, (2, 2),
        )
        return reg.normal_prior_loss(
            expected_surface, prior, torch.ones_like(expected_surface.valid),
        )

    vertices = base_vertices.clone().requires_grad_(True)
    analytic = torch.autograd.grad(evaluate(vertices), vertices)[0][0, 2]
    delta = 1e-5
    plus = base_vertices.clone()
    minus = base_vertices.clone()
    plus[0, 2] += delta
    minus[0, 2] -= delta
    numeric = (evaluate(plus) - evaluate(minus)) / (2.0 * delta)
    torch.testing.assert_close(analytic, numeric, rtol=2e-4, atol=2e-6)


def test_expected_surface_no_fragment_match_returns_connected_zero():
    dtype = torch.float64
    vertices = torch.tensor(
        [[-1.0, -1.0, 3.0], [1.0, -1.0, 3.0], [0.0, 1.0, 3.0]],
        dtype=dtype,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    frag_pix = torch.tensor([[0, 0, 0]], dtype=torch.int32)
    frag_attrs = torch.tensor([[0.25, 0.25, 0.0, 1.0]], dtype=dtype)
    frag_alpha = torch.tensor([0.5], dtype=dtype)
    K = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    Tcw = torch.eye(4, dtype=dtype).unsqueeze(0)
    pixels = torch.tensor([[0, 0, 1]], dtype=torch.int64)
    expected_surface = surface.vertex_expected_surface_samples(
        vertices,
        faces,
        frag_pix,
        frag_attrs,
        frag_alpha,
        K,
        Tcw,
        pixels,
        (1, 2),
    )

    assert not expected_surface.valid.any()
    torch.testing.assert_close(
        expected_surface.expected_camera_z, torch.zeros(1, dtype=dtype),
    )
    (
        expected_surface.expected_camera_z.sum()
        + expected_surface.rendered_normal_camera.sum()
    ).backward()
    assert vertices.grad is not None
    torch.testing.assert_close(vertices.grad, torch.zeros_like(vertices))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fragments_integrate_with_expected_surface_and_prior_losses():
    device = torch.device("cuda")
    dtype = torch.float32
    vertices = torch.tensor(
        [[-0.8, -0.8, 3.0], [0.8, -0.8, 3.0], [0.0, 0.8, 3.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    K = torch.tensor(
        [[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    Tcw = torch.eye(
        4, dtype=dtype, device=device,
    ).unsqueeze(0).requires_grad_(True)
    mvp = example_utils.mvp_from_K_Tcw(
        K, Tcw[0], (16, 16), z_near=0.1, z_far=100.0, flip_z=True,
    ).unsqueeze(0)
    vertices_clip = example_utils.project_vertices(vertices, mvp)
    alpha_src = torch.full(
        (1, 3, 1), 0.8, dtype=dtype, device=device, requires_grad=True,
    )
    _, fragments = ds.rasterize_multires_triangle_alpha(
        (16, 16),
        vertices_clip,
        faces,
        level=0,
        alpha_src=alpha_src,
        stochastic=False,
        return_fragments=True,
    )
    pixels = torch.unique(fragments.frag_pix.detach().to(torch.int64), dim=0)
    assert pixels.shape[0] > 0
    expected_surface = surface.vertex_expected_surface_samples(
        vertices,
        faces,
        fragments.frag_pix,
        fragments.frag_attrs,
        fragments.frag_alpha,
        K,
        Tcw,
        pixels,
        (16, 16),
    )
    assert expected_surface.valid.all()

    prior_depth = torch.full(
        expected_surface.expected_camera_z.shape,
        0.25,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    prior_normal = torch.tensor(
        [0.2, 0.0, -0.98], dtype=dtype, device=device,
    ).expand_as(expected_surface.rendered_normal_camera).clone().requires_grad_(True)
    prior_valid = torch.ones_like(expected_surface.valid)
    conditional_camera_z = (
        expected_surface.expected_camera_z
        / expected_surface.accumulated_opacity
    )
    torch.testing.assert_close(
        conditional_camera_z,
        torch.full_like(conditional_camera_z, 3.0),
        rtol=1e-5,
        atol=1e-5,
    )
    depth_loss = reg.inverse_depth_prior_loss(
        expected_surface,
        prior_depth,
        prior_valid,
    )
    torch.testing.assert_close(
        depth_loss,
        torch.tensor(
            abs(1.0 / (3.0 + 1e-6) - 0.25),
            dtype=dtype,
            device=device,
        ),
        rtol=1e-5,
        atol=1e-6,
    )
    loss = depth_loss + reg.normal_prior_loss(
        expected_surface, prior_normal, prior_valid,
    )
    loss.backward()

    assert vertices.grad is not None
    assert torch.isfinite(vertices.grad).all()
    assert vertices.grad.abs().sum() > 0
    assert alpha_src.grad is None
    assert K.grad is None
    assert Tcw.grad is None
    assert prior_depth.grad is None
    assert prior_normal.grad is None
