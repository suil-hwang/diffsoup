# python/diffsoup/remesh.py
"""Adaptive triangle-soup subdivision (world-space and clip-space)."""

import numpy as np
import torch
from . import _core


def split_triangle_soup(
    verts: torch.Tensor,   # (N, 3), float32
    faces: torch.Tensor,   # (M, 3), int32
    num_splits: int,
    tau: float = 0.0,
    *,
    return_vertex_provenance: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """Split a triangle soup by repeatedly bisecting the longest edge.

    Args:
        verts:      ``(N, 3)`` float32 vertex positions.
        faces:      ``(M, 3)`` int32 triangle indices.
        num_splits: Maximum number of edge bisections.  Use ``-1`` together
                    with ``tau > 0`` for threshold-only mode.
        tau:        Stop once the longest remaining edge ≤ ``tau``
                    (0 disables the threshold).
        return_vertex_provenance: Also return input source vertex indices and
            interpolation weights for every output vertex.

    Returns:
        The legacy result is ``(out_verts, out_faces, face_mapping,
        face_flags)``. If provenance is requested, appends
        ``vertex_source_indices`` and ``vertex_source_weights``, both shaped
        ``(N', 3)``. Each output vertex is their weighted sum over the input
        vertices; original inputs use ``(i, i, i)`` and ``(1, 0, 0)``.
    """
    assert verts.ndim == 2 and verts.shape[1] == 3, "verts must be (N, 3)"
    assert faces.ndim == 2 and faces.shape[1] == 3, "faces must be (M, 3)"

    dev = verts.device
    v_np = verts.float().detach().cpu().contiguous().numpy()
    f_np = faces.to(torch.int32).detach().cpu().contiguous().numpy()
    core_split = (
        _core.split_triangle_soup_with_provenance
        if return_vertex_provenance
        else _core.split_triangle_soup
    )
    outputs = core_split(v_np, f_np, int(num_splits), float(tau))

    result = (
        torch.from_numpy(outputs[0]).to(device=dev, dtype=torch.float32),
        torch.from_numpy(outputs[1]).to(device=dev, dtype=torch.int32),
        torch.from_numpy(outputs[2]).to(device=dev, dtype=torch.int32),
        torch.from_numpy(outputs[3]).to(device=dev, dtype=torch.int32),
    )
    if not return_vertex_provenance:
        return result
    return result + (
        torch.from_numpy(outputs[4]).to(device=dev, dtype=torch.int32),
        torch.from_numpy(outputs[5]).to(device=dev, dtype=torch.float32),
    )


def split_triangle_soup_until(
    verts: torch.Tensor,
    faces: torch.Tensor,
    tau: float,
    hard_cap: int | None = None,
    *,
    return_vertex_provenance: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """Split until all edges ≤ ``tau``.

    Args:
        verts:    ``(N, 3)`` float32 vertex positions.
        faces:    ``(M, 3)`` int32 triangle indices.
        tau:      Maximum allowed edge length.
        hard_cap: Optional upper bound on the number of bisections.
        return_vertex_provenance: Also return input source vertex indices and
            interpolation weights.

    Returns:
        Same optional four- or six-tuple as :func:`split_triangle_soup`.
    """
    ns = -1 if hard_cap is None else int(hard_cap)
    return split_triangle_soup(
        verts,
        faces,
        ns,
        tau=float(tau),
        return_vertex_provenance=return_vertex_provenance,
    )


def split_triangle_soup_clip(
    resolution: tuple[int, int],
    mvp: torch.Tensor,           # (4, 4)
    verts: torch.Tensor,         # (N, 3)
    faces: torch.Tensor,         # (M, 3)
    valid_faces: torch.Tensor,   # (M,)
    num_splits: int,
    tau_ratio: float = 0.0,
    *,
    return_vertex_provenance: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """Split a triangle soup based on image-space (clip-space) edge lengths.

    Pipeline:  world ``(N, 3)`` → clip ``(N, 4)`` via MVP → split in
    clip space → back to world ``(N', 3)``.

    Image-space length is measured in normalised device coordinates
    ``(x/w, y/w)`` with the x axis scaled by ``W/H`` so that the metric
    unit equals one image height.

    Edges with either endpoint outside the NDC cube ``[-1, 1]³`` are skipped.

    Args:
        resolution:  ``(H, W)`` image resolution.
        mvp:         ``(4, 4)`` model-view-projection matrix (row-major).
        verts:       ``(N, 3)`` float32 world-space positions.
        faces:       ``(M, 3)`` int32 triangle indices.
        valid_faces: ``(M,)`` int32 mask (1 = consider for splitting).
        num_splits:  Maximum bisections (``-1`` for threshold-only).
        tau_ratio:   Threshold in image-height units (0 disables).
        return_vertex_provenance: Also return input source vertex indices and
            interpolation weights for every output vertex.

    Returns:
        Same four-tuple as :func:`split_triangle_soup`.  If provenance is
        requested, appends ``vertex_source_indices`` and
        ``vertex_source_weights``, both shaped ``(N', 3)``. Each output vertex
        is the weighted sum of these original input vertices. Original input
        vertices have identity rows ``(i, i, i)`` with weights ``(1, 0, 0)``.
    """
    H, W = resolution
    assert mvp.shape == (4, 4)
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert valid_faces.shape == (faces.shape[0],)

    dev = verts.device

    verts = verts.float().contiguous()
    faces = faces.to(torch.int32).contiguous()
    valid_faces = valid_faces.to(torch.int32).contiguous()
    mvp = mvp.float().contiguous()

    aspect_wh = float(W) / float(H)
    core_split = (
        _core.split_triangle_soup_clip_with_provenance
        if return_vertex_provenance
        else _core.split_triangle_soup_clip
    )
    outputs = core_split(
        mvp.detach().cpu().numpy(),
        verts.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
        valid_faces.detach().cpu().numpy(),
        int(num_splits),
        float(tau_ratio),
        aspect_wh,
    )

    result = (
        torch.from_numpy(outputs[0]).to(device=dev, dtype=torch.float32),
        torch.from_numpy(outputs[1]).to(device=dev, dtype=torch.int32),
        torch.from_numpy(outputs[2]).to(device=dev, dtype=torch.int32),
        torch.from_numpy(outputs[3]).to(device=dev, dtype=torch.int32),
    )
    if not return_vertex_provenance:
        return result
    return result + (
        torch.from_numpy(outputs[4]).to(device=dev, dtype=torch.int32),
        torch.from_numpy(outputs[5]).to(device=dev, dtype=torch.float32),
    )


def split_triangle_soup_clip_until(
    resolution: tuple[int, int],
    mvp: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    valid_faces: torch.Tensor,
    tau_ratio: float,
    hard_cap: int | None = None,
    *,
    return_vertex_provenance: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """Split until all image-space edges ≤ ``tau_ratio``.

    Args:
        resolution:  ``(H, W)`` image resolution.
        mvp:         ``(4, 4)`` MVP matrix.
        verts:       ``(N, 3)`` float32 world-space positions.
        faces:       ``(M, 3)`` int32 triangle indices.
        valid_faces: ``(M,)`` int32 mask.
        tau_ratio:   Threshold in image-height units.
        hard_cap:    Optional upper bound on the number of bisections.
        return_vertex_provenance: Also return input source vertex indices and
            interpolation weights.

    Returns:
        Same optional four- or six-tuple as
        :func:`split_triangle_soup_clip`.
    """
    ns = -1 if hard_cap is None else int(hard_cap)
    return split_triangle_soup_clip(
        resolution, mvp, verts, faces, valid_faces, ns, tau_ratio,
        return_vertex_provenance=return_vertex_provenance,
    )


def expand_by_index(
    source: torch.Tensor,
    index_map: torch.Tensor,
) -> torch.Tensor:
    """Gather rows from ``source`` according to ``index_map``.

    Useful for propagating per-face features through a subdivision step:
    if face *j* in the output descends from face *i* in the input, set
    ``index_map[j] = i`` and call this function to copy the features.

    Args:
        source:    ``(N, ...)`` tensor of parent features.
        index_map: ``(N',)`` long tensor of indices into ``[0, N)``.

    Returns:
        ``(N', ...)`` tensor with ``result[j] = source[index_map[j]]``.
    """
    if not torch.is_tensor(source):
        raise TypeError("`source` must be a torch.Tensor.")
    if not torch.is_tensor(index_map):
        raise TypeError("`index_map` must be a torch.Tensor.")

    if index_map.dtype != torch.long:
        index_map = index_map.to(torch.long)
    index_map = index_map.to(source.device)

    N = source.size(0)
    if torch.any((index_map < 0) | (index_map >= N)):
        raise ValueError("`index_map` contains out-of-range indices.")

    return source.index_select(0, index_map)
