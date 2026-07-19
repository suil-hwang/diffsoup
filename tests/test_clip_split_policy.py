"""Synthetic regressions for the clip-space edge-splitting policy."""

from __future__ import annotations

import pytest
import torch


ds = pytest.importorskip("diffsoup")

pytestmark = pytest.mark.native


_CASES = [
    ("fully_inside", "identity", (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), True),
    ("one_outside_crossing", "identity", (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), False),
    ("both_outside_crossing", "identity", (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), False),
    ("fully_outside", "identity", (2.0, 0.0, 0.0), (3.0, 0.0, 0.0), False),
    ("near_crossing", "identity", (0.0, 0.0, -2.0), (0.5, 0.0, 0.0), False),
    ("far_crossing", "identity", (0.0, 0.0, 2.0), (0.5, 0.0, 0.0), False),
    ("behind_to_inside", "w_from_z", (0.0, 0.0, -1.0), (0.5, 0.0, 1.0), False),
    ("both_behind", "w_from_z", (0.0, 0.0, -1.0), (0.5, 0.0, -0.5), False),
    ("exact_boundary", "identity", (-1.0, -0.2, 0.0), (1.0, 0.2, 0.0), True),
    ("degenerate_w_zero", "w_from_z", (0.0, 0.0, 0.0), (0.5, 0.0, 1.0), False),
    ("tiny_positive_w", "w_from_z", (0.0, 0.0, 1e-8), (0.5, 0.0, 1.0), True),
]


def _mvp(kind: str) -> torch.Tensor:
    if kind == "identity":
        return torch.eye(4, dtype=torch.float32)
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _inside_ndc(mvp: torch.Tensor, point: tuple[float, float, float]) -> bool:
    homogeneous = mvp @ torch.tensor((*point, 1.0), dtype=torch.float32)
    w = homogeneous[3]
    return bool(w > 0.0 and torch.all(homogeneous[:3].abs() <= w))


@pytest.mark.parametrize(
    ("name", "mvp_kind", "a", "b", "expected_split"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_clip_split_requires_both_edge_endpoints_inside_ndc(
    name: str,
    mvp_kind: str,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    expected_split: bool,
) -> None:
    """Crossing the frustum is insufficient unless both endpoints are inside."""
    del name
    mvp = _mvp(mvp_kind)
    vertices = torch.tensor([a, b, a], dtype=torch.float32)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    valid_faces = torch.ones(1, dtype=torch.int32)

    out_vertices, out_faces, _, _ = ds.split_triangle_soup_clip(
        (100, 100),
        mvp,
        vertices,
        faces,
        valid_faces,
        num_splits=1,
        tau_ratio=0.0,
    )

    endpoint_policy = _inside_ndc(mvp, a) and _inside_ndc(mvp, b)
    native_split = out_faces.shape[0] > faces.shape[0]
    assert endpoint_policy is expected_split
    assert native_split is expected_split
    assert out_vertices.shape == ((6, 3) if expected_split else (3, 3))
    assert out_faces.shape == ((2, 3) if expected_split else (1, 3))
