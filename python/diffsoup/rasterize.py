# python/diffsoup/rasterize.py
"""Software rasterisation, edge-gradient computation, and view-direction encoding."""

import torch
from typing import NamedTuple, Tuple

from . import _core


def _cuda_stream(tensor: torch.Tensor) -> int:
    """Return PyTorch's current CUDA stream handle for the tensor device."""
    return torch.cuda.current_stream(tensor.device).cuda_stream


# ---------------------------------------------------------------------------
#  Fragment computation
# ---------------------------------------------------------------------------

def _filter_valid_fragments(
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compact fragment buffers by removing invalid (off-screen) entries.

    Args:
        frag_pix:   (N, 3) int32 — (batch, h, w) per fragment.
        frag_attrs: (N, 4) float32 — (bary0, bary1, z, triangle_id+1).

    Returns:
        Compacted ``frag_pix`` and ``frag_attrs`` tensors containing only
        valid fragments.
    """
    counter = torch.empty(1, dtype=torch.int32, device=frag_pix.device)
    stream = _cuda_stream(frag_pix)
    valid_count = _core.count_valid_fragments(
        frag_pix, counter, stream,
    )
    frag_pix_out = torch.empty(
        (valid_count, 3), dtype=torch.int32, device=frag_pix.device
    )
    frag_attrs_out = torch.empty(
        (valid_count, 4), dtype=torch.float32, device=frag_attrs.device
    )
    _core.compact_valid_fragments(
        frag_pix, frag_attrs, frag_pix_out, frag_attrs_out,
        counter, stream,
    )
    return frag_pix_out, frag_attrs_out


def _compute_fragments(
    resolution: Tuple[int, int],
    pos: torch.Tensor,   # (B, V, 4)
    tri: torch.Tensor,   # (T, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterise a triangle mesh into per-pixel fragments.

    Args:
        resolution: Image resolution ``(H, W)``.
        pos:  Homogeneous vertex positions ``(B, V, 4)``, float32 CUDA.
        tri:  Triangle indices ``(T, 3)``, int32 CUDA.

    Returns:
        frag_pix:   (N, 3) int32 — ``(batch, h, w)`` per valid fragment.
        frag_attrs: (N, 4) float32 — ``(bary0, bary1, z, triangle_id+1)``.
    """
    H, W = resolution
    B, V, _ = pos.shape
    T, _ = tri.shape
    device = pos.device

    assert pos.shape == (B, V, 4) and pos.dtype == torch.float32 and pos.is_contiguous()
    assert tri.shape == (T, 3) and tri.dtype == torch.int32 and tri.is_contiguous()
    assert pos.is_cuda and tri.device == device

    rects = torch.empty((B * T, 4), dtype=torch.int32, device=device)
    frag_prefix = torch.empty(B * T, dtype=torch.int32, device=device)
    triangle_stats = torch.empty(2, dtype=torch.int32, device=device)
    stream = _cuda_stream(pos)
    num_frags, active_triangles, max_candidates = _core.compute_triangle_rects(
        H, W, pos, tri, rects, frag_prefix, triangle_stats, stream
    )

    frag_pix = torch.empty((num_frags, 3), dtype=torch.int32, device=device)
    frag_attrs = torch.empty((num_frags, 4), dtype=torch.float32, device=device)

    _core.compute_fragments(
        H, W, pos, tri, frag_prefix, rects, frag_pix, frag_attrs,
        active_triangles, max_candidates, stream,
    )

    return _filter_valid_fragments(frag_pix, frag_attrs)


class PixelFragmentCSR(NamedTuple):
    """Pixel-to-fragment compressed sparse row index."""

    pixel_offsets: torch.Tensor
    fragment_indices: torch.Tensor


def build_pixel_fragment_csr(
    frag_pix: torch.Tensor,
    batch_size: int,
    image_size: Tuple[int, int],
) -> PixelFragmentCSR:
    """Build an opt-in pixel-to-fragment CSR index on the current stream.

    ``pixel_offsets`` has ``batch_size * H * W + 1`` entries; the fragments
    for flattened pixel ``p`` are ``fragment_indices[offsets[p]:offsets[p+1]]``.
    Invalid fragment rows are omitted, and order inside each segment is not
    defined.
    """
    height, width = image_size
    assert frag_pix.ndim == 2 and frag_pix.shape[-1] == 3
    assert frag_pix.dtype == torch.int32 and frag_pix.is_contiguous()
    assert frag_pix.is_cuda
    assert batch_size >= 0 and height > 0 and width > 0
    total_pixels = batch_size * height * width
    int32_max = torch.iinfo(torch.int32).max
    assert total_pixels < int32_max and frag_pix.shape[0] <= int32_max

    pixel_offsets = torch.empty(
        total_pixels + 1, dtype=torch.int32, device=frag_pix.device,
    )
    pixel_cursors = torch.empty(
        total_pixels, dtype=torch.int32, device=frag_pix.device,
    )
    fragment_indices = torch.empty(
        frag_pix.shape[0], dtype=torch.int32, device=frag_pix.device,
    )
    _core.build_pixel_fragment_csr(
        batch_size, height, width, frag_pix,
        pixel_offsets, pixel_cursors, fragment_indices,
        _cuda_stream(frag_pix),
    )
    return PixelFragmentCSR(pixel_offsets, fragment_indices)


# ---------------------------------------------------------------------------
#  Depth test
# ---------------------------------------------------------------------------

def _depth_test(
    resolution: Tuple[int, int],
    pos: torch.Tensor,
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor,
    frag_alpha: torch.Tensor,
    alpha_thresh: torch.Tensor,
) -> torch.Tensor:
    """Resolve fragment visibility via depth testing.

    Fragments whose alpha falls below ``alpha_thresh`` are discarded before
    the depth comparison.

    Args:
        resolution:   Image resolution ``(H, W)``.
        pos:          Homogeneous vertex positions ``(B, V, 4)``, float32 CUDA.
        frag_pix:     ``(N, 3)`` int32 — ``(batch, h, w)`` per fragment.
        frag_attrs:   ``(N, 4)`` float32 — ``(bary0, bary1, z, triangle_id+1)``.
        frag_alpha:   ``(N,)`` float32 — per-fragment opacity.
        alpha_thresh: ``(N,)`` float32 — per-fragment stochastic threshold.

    Returns:
        rast_out: ``(B, H, W, 4)`` float32 CUDA.  Each pixel stores
        ``(bary0, bary1, z, triangle_id+1)``; background pixels are zero.
    """
    H, W = resolution
    B, V, _ = pos.shape
    num_frags, _ = frag_pix.shape
    device = pos.device

    assert pos.shape == (B, V, 4) and pos.dtype == torch.float32 and pos.is_contiguous()
    assert frag_pix.shape == (num_frags, 3) and frag_pix.dtype == torch.int32 and frag_pix.is_contiguous()
    assert frag_attrs.shape == (num_frags, 4) and frag_attrs.dtype == torch.float32 and frag_attrs.is_contiguous()
    assert frag_alpha.shape == (num_frags,) and frag_alpha.dtype == torch.float32 and frag_alpha.is_contiguous()
    assert alpha_thresh.shape == (num_frags,) and alpha_thresh.dtype == torch.float32 and alpha_thresh.is_contiguous()
    assert pos.is_cuda
    assert frag_pix.device == device
    assert frag_attrs.device == device
    assert frag_alpha.device == device
    assert alpha_thresh.device == device

    frag_index = torch.empty(B, H, W, dtype=torch.int64, device=device)
    rast = torch.empty(B, H, W, 4, dtype=torch.float32, device=device)
    _core.depth_test(
        frag_pix, frag_attrs, frag_alpha, alpha_thresh,
        frag_index, rast, _cuda_stream(pos),
    )
    return rast


def _depth_test_counter_rng(
    resolution: Tuple[int, int],
    pos: torch.Tensor,
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor,
    frag_alpha: torch.Tensor,
    rng_seed: int,
    rng_counter: int,
) -> torch.Tensor:
    """Resolve visibility with stateless per-fragment Philox thresholds."""
    H, W = resolution
    B, V, _ = pos.shape
    num_frags, _ = frag_pix.shape
    device = pos.device

    assert pos.shape == (B, V, 4) and pos.dtype == torch.float32 and pos.is_contiguous()
    assert frag_pix.shape == (num_frags, 3) and frag_pix.dtype == torch.int32 and frag_pix.is_contiguous()
    assert frag_attrs.shape == (num_frags, 4) and frag_attrs.dtype == torch.float32 and frag_attrs.is_contiguous()
    assert frag_alpha.shape == (num_frags,) and frag_alpha.dtype == torch.float32 and frag_alpha.is_contiguous()
    assert pos.is_cuda
    assert frag_pix.device == device
    assert frag_attrs.device == device
    assert frag_alpha.device == device
    assert 0 <= rng_seed < 1 << 64
    assert 0 <= rng_counter < 1 << 64

    frag_index = torch.empty(B, H, W, dtype=torch.int64, device=device)
    rast = torch.empty(B, H, W, 4, dtype=torch.float32, device=device)
    _core.depth_test_counter_rng(
        frag_pix, frag_attrs, frag_alpha, rng_seed, rng_counter,
        frag_index, rast, _cuda_stream(pos),
    )
    return rast


# ---------------------------------------------------------------------------
#  Stochastic opacity masking
# ---------------------------------------------------------------------------
#
#  The forward pass returns a **zero-valued** scalar.  Its only purpose is to
#  register the analytic gradient of the stochastic-opacity-masking auxiliary
#  objective into the autograd graph, so that ``loss.backward()`` propagates
#  the correct signal through ``alpha_src``.
# ---------------------------------------------------------------------------

class _OpacityAuxLossFn(torch.autograd.Function):
    """Autograd hook for the stochastic opacity masking auxiliary gradient.

    The forward value is identically zero; all useful work happens in the
    backward pass, which computes ∂L_aux/∂alpha_src analytically via the
    CUDA kernels.
    """

    @staticmethod
    def forward(
        ctx,
        color: torch.Tensor,       # (B, H, W, C)
        target: torch.Tensor,      # (B, H, W, C)
        rast: torch.Tensor,        # (B, H, W, 4)
        level: int,
        alpha_src: torch.Tensor,
        frag_pix: torch.Tensor,
        frag_attrs: torch.Tensor,
        frag_alpha: torch.Tensor,
    ) -> torch.Tensor:
        dev = rast.device
        assert alpha_src.ndim == 2

        grad_frag_alpha = torch.empty_like(frag_alpha)
        _core.backward_opacity_aux_loss(
            color, target, rast, frag_pix, frag_attrs,
            frag_alpha, grad_frag_alpha, _cuda_stream(rast),
        )

        grad_alpha_src = torch.zeros_like(alpha_src)
        _core.backward_multires_triangle_alpha(
            frag_attrs, level, level,
            grad_alpha_src, grad_frag_alpha, _cuda_stream(rast),
        )

        weight = 1.0 / color.numel()
        ctx.save_for_backward(
            grad_alpha_src,
            torch.tensor([weight], dtype=torch.float32),
        )

        # Identically-zero scalar — the gradient is the whole point.
        return torch.zeros(1, dtype=torch.float32, device=dev)

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor):
        grad_alpha_src, weight = ctx.saved_tensors
        grad_alpha_src = weight.item() * grad_loss * grad_alpha_src
        return None, None, None, None, grad_alpha_src, None, None, None


def opacity_aux_loss(
    color: torch.Tensor,       # (B, H, W, C)
    target: torch.Tensor,      # (B, H, W, C)
    rast: torch.Tensor,        # (B, H, W, 4)
    pos: torch.Tensor,         # (B, V, 4)
    tri: torch.Tensor,         # (T, 3)
    level: int,
    alpha_src: torch.Tensor,
    fragments=None,
) -> torch.Tensor:
    """Stochastic opacity masking auxiliary loss (zero-valued gradient hook).

    This function returns a **scalar that is identically zero**.  Its sole
    purpose is to inject the analytic gradient of the opacity-masking
    objective into the autograd tape so that calling ``loss.backward()``
    propagates the correct signal into ``alpha_src``.

    Typical usage::

        aux = diffsoup.opacity_aux_loss(color, target, rast, pos, tri, level, alpha)
        loss = mse_loss + aux          # aux.item() == 0
        loss.backward()                # gradients flow into alpha via aux

    Args:
        color:     Current rendering ``(B, H, W, C)``, float32 CUDA.
        target:    Target image ``(B, H, W, C)``, float32 CUDA.
        rast:      Rasterisation output ``(B, H, W, 4)``, float32 CUDA.
        pos:       Homogeneous vertex positions ``(B, V, 4)``, float32 CUDA.
        tri:       Triangle indices ``(T, 3)``, int32 CUDA.
        level:     Multi-resolution level (≥ 0).
        alpha_src: Per-triangle opacity features ``(T, S, 1)``, float32 CUDA,
                   where ``S = feats_at_level(level)``.
        fragments: Optional fragment intermediates returned by
                   ``rasterize_multires_triangle_alpha(..., return_fragments=True)``.
                   When omitted, fragments are recomputed for compatibility.

    Returns:
        A zero-valued float32 CUDA scalar with an autograd backward that
        populates ``alpha_src.grad``.
    """
    B, H, W, C = color.shape
    dev = color.device
    _, V, _ = pos.shape
    T, _ = tri.shape
    _, S, _ = alpha_src.shape

    assert color.is_contiguous() and color.dtype == torch.float32
    assert target.shape == (B, H, W, C) and target.is_contiguous() and target.dtype == torch.float32
    assert rast.shape == (B, H, W, 4) and rast.is_contiguous() and rast.dtype == torch.float32
    assert pos.shape == (B, V, 4) and pos.is_contiguous() and pos.dtype == torch.float32
    assert tri.shape == (T, 3) and tri.is_contiguous() and tri.dtype == torch.int32
    assert alpha_src.shape == (T, S, 1) and alpha_src.dtype == torch.float32 and alpha_src.is_contiguous()
    assert color.is_cuda
    assert target.device == dev and rast.device == dev
    assert pos.device == dev and tri.device == dev

    alpha_src = alpha_src.squeeze(-1)
    if fragments is None:
        frag_pix, frag_attrs = _compute_fragments((H, W), pos, tri)
        frag_alpha = torch.empty(frag_pix.shape[0], dtype=torch.float32, device=dev)
        _core.multires_triangle_alpha(
            frag_attrs, level, level, alpha_src, frag_alpha,
            _cuda_stream(rast),
        )
    else:
        if len(fragments) != 3:
            raise ValueError("fragments must contain frag_pix, frag_attrs, and frag_alpha")
        frag_pix, frag_attrs, frag_alpha = fragments
        num_frags = frag_pix.shape[0]
        assert frag_pix.shape == (num_frags, 3) and frag_pix.dtype == torch.int32
        assert frag_attrs.shape == (num_frags, 4) and frag_attrs.dtype == torch.float32
        assert frag_alpha.shape == (num_frags,) and frag_alpha.dtype == torch.float32
        assert frag_pix.is_contiguous() and frag_attrs.is_contiguous() and frag_alpha.is_contiguous()
        assert frag_pix.device == dev and frag_attrs.device == dev and frag_alpha.device == dev

    return _OpacityAuxLossFn.apply(
        color, target, rast, level, alpha_src,
        frag_pix, frag_attrs, frag_alpha,
    )


# ---------------------------------------------------------------------------
#  Edge-gradient pass
# ---------------------------------------------------------------------------

class _EdgeGradFn(torch.autograd.Function):
    """Inject silhouette / edge gradients into vertex positions."""

    @staticmethod
    def forward(ctx, color, rast, pos, tri):
        ctx.save_for_backward(color, rast, pos, tri)
        return color

    @staticmethod
    def backward(ctx, grad_color):
        color, rast, pos, tri = ctx.saved_tensors
        grad_color = grad_color.contiguous()
        grad_pos = torch.zeros_like(pos)

        _core.backward_edge_grad(
            color, grad_color, rast, pos, grad_pos, tri, _cuda_stream(color)
        )
        return grad_color, None, grad_pos, None


def edge_grad(
    color: torch.Tensor,   # (B, H, W, C)
    rast: torch.Tensor,    # (B, H, W, 4)
    pos: torch.Tensor,     # (B, V, 4)
    tri: torch.Tensor,     # (T, 3)
) -> torch.Tensor:
    """Attach silhouette edge gradients to vertex positions.

    Wraps ``color`` in an autograd function whose backward pass computes
    ∂color/∂pos along triangle silhouette edges, enabling gradient-based
    optimisation of mesh geometry through rasterisation boundaries.

    Args:
        color: Current rendering ``(B, H, W, C)``, float32 CUDA.
        rast:  Rasterisation output ``(B, H, W, 4)``, float32 CUDA.
        pos:   Homogeneous vertex positions ``(B, V, 4)``, float32 CUDA.
        tri:   Triangle indices ``(T, 3)``, int32 CUDA.

    Returns:
        The same ``color`` tensor, now carrying an autograd backward that
        populates ``pos.grad`` with edge-aware gradients.
    """
    B, H, W, C = color.shape
    _, V, _ = pos.shape
    T, _ = tri.shape

    assert color.is_contiguous() and color.dtype == torch.float32
    assert rast.shape == (B, H, W, 4) and rast.is_contiguous() and rast.dtype == torch.float32
    assert pos.shape == (B, V, 4) and pos.is_contiguous() and pos.dtype == torch.float32
    assert tri.shape == (T, 3) and tri.is_contiguous() and tri.dtype == torch.int32
    assert color.is_cuda
    assert rast.device == color.device
    assert pos.device == color.device
    assert tri.device == color.device

    return _EdgeGradFn.apply(color, rast, pos, tri)


# ---------------------------------------------------------------------------
#  View-direction encodings
# ---------------------------------------------------------------------------

def encode_view_dir_sh2(
    rast: torch.Tensor,      # (B, H, W, 4)
    inv_mvp: torch.Tensor,   # (B, 4, 4)
) -> torch.Tensor:
    """Evaluate order-2 spherical-harmonic basis on per-pixel view directions.

    Args:
        rast:    Rasterisation output ``(B, H, W, 4)``, float32 CUDA.
        inv_mvp: Inverse MVP transforms ``(B, 4, 4)``, float32 CUDA.

    Returns:
        encoding: ``(B, H, W, 9)`` float32 CUDA — the 9 SH2 coefficients.
    """
    B, H, W, _ = rast.shape
    dev = rast.device

    assert rast.shape == (B, H, W, 4) and rast.is_contiguous() and rast.dtype == torch.float32
    assert inv_mvp.shape == (B, 4, 4) and inv_mvp.is_contiguous() and inv_mvp.dtype == torch.float32
    assert rast.is_cuda and inv_mvp.device == dev

    encoding = torch.empty(B, H, W, 9, dtype=torch.float32, device=dev)
    _core.encode_view_dir_sh2(rast, inv_mvp, encoding, _cuda_stream(rast))
    return encoding


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def count_triangle_ids(
    rast: torch.Tensor,
    num_tris: int,
) -> torch.Tensor:
    """Count how many pixels each triangle covers in a rasterisation buffer.

    Args:
        rast:     ``(B, H, W, 4)`` float32 CUDA — the last channel stores
                  1-based triangle IDs (0 = background).
        num_tris: Total number of triangles in the mesh.

    Returns:
        ``(num_tris,)`` long tensor of per-triangle pixel counts.
    """
    tri_ids = rast[..., -1].long()
    tri_ids = tri_ids[tri_ids > 0] - 1
    count = torch.bincount(tri_ids, minlength=num_tris)
    assert count.shape[0] == num_tris
    return count
