"""CUDA regressions for raster fragments and accumulation-plan kernels."""

import pytest
import torch


ds = pytest.importorskip("diffsoup")
_multires = pytest.importorskip("diffsoup.multires")
_core = pytest.importorskip("diffsoup._core")


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="DiffSoup CUDA optimization tests require a CUDA device",
    ),
]


def _target_lattice_raster(num_triangles: int, level: int) -> torch.Tensor:
    num_samples = ds.feats_at_level(level)
    sample = torch.arange(num_samples, dtype=torch.int64, device="cuda")
    sample_float = sample.float()
    diagonal = torch.floor(
        (torch.sqrt(8.0 * sample_float + 1.0) - 1.0) * 0.5
    ).to(torch.int64)
    y = sample - diagonal * (diagonal + 1) // 2
    x = diagonal - y

    rast = torch.zeros(
        (1, num_triangles, num_samples, 4),
        dtype=torch.float32,
        device="cuda",
    )
    resolution = float(1 << level)
    rast[0, :, :, 0] = (x.float() / resolution)[None, :]
    rast[0, :, :, 1] = (y.float() / resolution)[None, :]
    rast[0, :, :, 3] = torch.arange(
        1, num_triangles + 1, dtype=torch.float32, device="cuda"
    )[:, None]
    return rast


@pytest.mark.parametrize(
    ("min_level", "max_level", "target_level", "feature_dim"),
    [
        (0, 0, 0, 1),
        (0, 0, 2, 7),
        (2, 5, 5, 1),
        (2, 5, 5, 7),
        (2, 5, 3, 7),
        (2, 5, 0, 7),
    ],
)
def test_accumulation_matches_interpolation_reference(
    min_level: int,
    max_level: int,
    target_level: int,
    feature_dim: int,
):
    torch.manual_seed(7)
    num_triangles = 5
    num_features = sum(
        ds.feats_at_level(level)
        for level in range(min_level, max_level + 1)
    )
    num_target = ds.feats_at_level(target_level)
    features = torch.randn(
        (num_triangles, num_features, feature_dim),
        dtype=torch.float32,
        device="cuda",
    )
    rast = _target_lattice_raster(num_triangles, target_level)
    stream = torch.cuda.current_stream().cuda_stream

    expected = torch.empty(
        (1, num_triangles, num_target, feature_dim),
        dtype=torch.float32,
        device="cuda",
    )
    _core.multires_triangle_color(
        rast,
        min_level,
        max_level,
        features,
        expected,
        stream,
    )
    actual = ds.accumulate_to_level(
        min_level,
        max_level,
        features,
        target_level,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)

    grad_output = torch.randn_like(actual)
    expected_grad = torch.zeros_like(features)
    _core.backward_multires_triangle_color(
        rast,
        min_level,
        max_level,
        expected_grad,
        grad_output.unsqueeze(0),
        stream,
    )

    differentiable_features = features.detach().clone().requires_grad_(True)
    output = ds.accumulate_to_level(
        min_level,
        max_level,
        differentiable_features,
        target_level,
    )
    output.backward(grad_output)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        differentiable_features.grad,
        expected_grad,
        rtol=1e-5,
        atol=3e-5,
    )


def _compact_fragment_rows(
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    stream = torch.cuda.current_stream(frag_pix.device)
    counter = torch.empty(1, dtype=torch.int32, device=frag_pix.device)
    count = _core.count_valid_fragments(
        frag_pix, counter, stream.cuda_stream
    )
    pix_out = torch.full(
        (count, 3), -9999, dtype=torch.int32, device=frag_pix.device
    )
    attrs_out = torch.full(
        (count, 4), -9999.0, dtype=torch.float32, device=frag_attrs.device
    )
    _core.compact_valid_fragments(
        frag_pix,
        frag_attrs,
        pix_out,
        attrs_out,
        counter,
        stream.cuda_stream,
    )
    return count, counter, pix_out, attrs_out


def _fragment_compaction_inputs(
    num_fragments: int,
    pattern: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.arange(num_fragments, dtype=torch.int32, device="cuda")
    if pattern == "all":
        valid = torch.ones(num_fragments, dtype=torch.bool, device="cuda")
    elif pattern == "none":
        valid = torch.zeros(num_fragments, dtype=torch.bool, device="cuda")
    else:
        valid = (source.remainder(5) != 1)
        if num_fragments:
            valid[-1] = True

    frag_pix = torch.stack(
        (
            torch.where(valid, source.remainder(3), -torch.ones_like(source)),
            source * 2 + 3,
            source * 3 + 5,
        ),
        dim=1,
    ).contiguous()
    source_float = source.float()
    frag_attrs = torch.stack(
        (
            source_float + 0.25,
            source_float + 0.5,
            source_float + 0.75,
            source_float + 1.0,
        ),
        dim=1,
    ).contiguous()
    return frag_pix, frag_attrs, valid


def _assert_fragment_compaction(
    frag_pix: torch.Tensor,
    frag_attrs: torch.Tensor,
    valid: torch.Tensor,
    count: int,
    counter: torch.Tensor,
    pix_out: torch.Tensor,
    attrs_out: torch.Tensor,
) -> None:
    expected = torch.nonzero(valid, as_tuple=False).flatten()
    assert count == expected.numel()
    assert pix_out.shape == (count, 3)
    assert attrs_out.shape == (count, 4)
    if frag_pix.shape[0]:
        assert counter.item() == count

    source_rows = attrs_out[:, 3].to(torch.int64) - 1
    torch.testing.assert_close(torch.sort(source_rows).values, expected)
    torch.testing.assert_close(pix_out, frag_pix.index_select(0, source_rows))
    torch.testing.assert_close(attrs_out, frag_attrs.index_select(0, source_rows))

    block_ids = source_rows // 256
    if block_ids.numel():
        run_starts = torch.ones_like(block_ids, dtype=torch.bool)
        run_starts[1:] = block_ids[1:] != block_ids[:-1]
        runs = block_ids[run_starts]
        assert runs.unique().numel() == runs.numel()
        for block_id in runs.tolist():
            local_rows = source_rows[block_ids == block_id]
            assert bool((local_rows[1:] > local_rows[:-1]).all())


@pytest.mark.parametrize(
    ("num_fragments", "pattern"),
    [
        (0, "mixed"),
        (1, "none"),
        (1, "all"),
        (31, "mixed"),
        (32, "mixed"),
        (33, "mixed"),
        (255, "mixed"),
        (256, "mixed"),
        (257, "mixed"),
        (513, "none"),
        (513, "all"),
        (513, "mixed"),
        (4099, "mixed"),
    ],
)
def test_fragment_count_and_compaction_preserve_rows_and_pairing(
    num_fragments: int,
    pattern: str,
):
    frag_pix, frag_attrs, valid = _fragment_compaction_inputs(
        num_fragments, pattern
    )
    result = _compact_fragment_rows(frag_pix, frag_attrs)
    _assert_fragment_compaction(frag_pix, frag_attrs, valid, *result)


def test_fragment_count_and_compaction_use_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        frag_pix, frag_attrs, valid = _fragment_compaction_inputs(4099, "mixed")
        result = _compact_fragment_rows(frag_pix, frag_attrs)
    stream.synchronize()
    _assert_fragment_compaction(frag_pix, frag_attrs, valid, *result)


def _single_triangle_inputs():
    positions = torch.tensor(
        [[
            [-0.8, -0.8, 0.0, 1.0],
            [0.8, -0.8, 0.0, 1.0],
            [0.0, 0.8, 0.0, 1.0],
        ]],
        dtype=torch.float32,
        device="cuda",
    )
    triangles = torch.tensor(
        [[0, 1, 2]],
        dtype=torch.int32,
        device="cuda",
    )
    alpha = torch.full(
        (1, 3, 1),
        0.8,
        dtype=torch.float32,
        device="cuda",
    )
    return positions, triangles, alpha


def test_fragment_reuse_preserves_auxiliary_gradient_and_zero_background():
    torch.manual_seed(11)
    positions, triangles, alpha = _single_triangle_inputs()
    rast, fragments = ds.rasterize_multires_triangle_alpha(
        (64, 64),
        positions,
        triangles,
        level=0,
        alpha_src=alpha,
        stochastic=False,
        return_fragments=True,
    )
    mask = rast[..., 3] > 0
    assert mask.any()
    assert (~mask).any()
    assert torch.count_nonzero(rast[~mask]) == 0

    features = torch.randn((1, 3, 7), dtype=torch.float32, device="cuda")
    color_features = ds.multires_triangle_color(rast, 0, features)
    inv_mvp = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0)
    encoding = ds.encode_view_dir_sh2(rast, inv_mvp)
    assert torch.count_nonzero(color_features[~mask]) == 0
    assert torch.count_nonzero(encoding[~mask]) == 0

    color = torch.rand((1, 64, 64, 3), device="cuda")
    target = torch.rand_like(color)

    cached_alpha = alpha.clone().requires_grad_(True)
    cached_loss = ds.opacity_aux_loss(
        color,
        target,
        rast,
        positions,
        triangles,
        level=0,
        alpha_src=cached_alpha,
        fragments=fragments,
    )
    cached_loss.backward()

    fallback_alpha = alpha.clone().requires_grad_(True)
    fallback_loss = ds.opacity_aux_loss(
        color,
        target,
        rast,
        positions,
        triangles,
        level=0,
        alpha_src=fallback_alpha,
    )
    fallback_loss.backward()
    torch.cuda.synchronize()

    assert cached_loss.item() == 0.0
    assert fallback_loss.item() == 0.0
    torch.testing.assert_close(
        cached_alpha.grad,
        fallback_alpha.grad,
        rtol=1e-6,
        atol=1e-7,
    )


def test_empty_geometry_produces_zero_raster():
    positions = torch.empty((1, 0, 4), dtype=torch.float32, device="cuda")
    triangles = torch.empty((0, 3), dtype=torch.int32, device="cuda")
    alpha = torch.empty((0, ds.feats_at_level(0), 1), device="cuda")

    rast, fragments = ds.rasterize_multires_triangle_alpha(
        (16, 24),
        positions,
        triangles,
        level=0,
        alpha_src=alpha,
        stochastic=False,
        return_fragments=True,
    )

    assert rast.shape == (1, 16, 24, 4)
    assert torch.count_nonzero(rast) == 0
    assert fragments.frag_pix.shape == (0, 3)
    assert fragments.frag_attrs.shape == (0, 4)
    assert fragments.frag_alpha.shape == (0,)


def test_rasterization_on_nondefault_stream_matches_default_stream():
    positions, triangles, alpha = _single_triangle_inputs()
    expected = ds.rasterize_multires_triangle_alpha(
        (64, 64),
        positions,
        triangles,
        level=0,
        alpha_src=alpha,
        stochastic=False,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = ds.rasterize_multires_triangle_alpha(
            (64, 64),
            positions,
            triangles,
            level=0,
            alpha_src=alpha,
            stochastic=False,
        )
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_accumulation_plan_reuse_on_nondefault_stream():
    torch.manual_seed(19)
    min_level, max_level, target_level = 2, 5, 5
    num_features = sum(
        ds.feats_at_level(level)
        for level in range(min_level, max_level + 1)
    )
    features = torch.randn(
        (3, num_features, 7), dtype=torch.float32, device="cuda"
    )
    grad_output = torch.randn(
        (3, ds.feats_at_level(target_level), 7),
        dtype=torch.float32,
        device="cuda",
    )

    expected_features = features.clone().requires_grad_(True)
    expected = ds.accumulate_to_level(
        min_level, max_level, expected_features, target_level
    )
    expected.backward(grad_output)
    torch.cuda.synchronize()

    stream_features = features.clone().requires_grad_(True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = ds.accumulate_to_level(
            min_level, max_level, stream_features, target_level
        )
        actual.backward(grad_output)
    stream.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        stream_features.grad, expected_features.grad, rtol=1e-5, atol=3e-5
    )


def _native_accumulation_backward_gather(
    min_level: int,
    max_level: int,
    target_level: int,
    grad_target: torch.Tensor,
) -> torch.Tensor:
    """Run the production reverse-CSR gather without the legacy scatter kernel."""
    plan = _multires._get_accumulation_plan(
        min_level, max_level, target_level, grad_target.device
    )
    source_size = sum(
        ds.feats_at_level(level)
        for level in range(min_level, max_level + 1)
    )
    grad_gather = torch.empty(
        grad_target.shape[0],
        source_size,
        grad_target.shape[2],
        dtype=torch.float32,
        device=grad_target.device,
    )
    stream = torch.cuda.current_stream(grad_target.device).cuda_stream
    _core.accumulate_to_level_backward_gather(
        min_level,
        max_level,
        target_level,
        grad_gather,
        grad_target,
        plan.reverse_offsets,
        plan.reverse_target_indices,
        plan.reverse_weights,
        stream,
    )
    return grad_gather


@pytest.mark.parametrize(
    ("min_level", "max_level", "target_level", "feature_dim"),
    [
        (0, 0, 0, 1),
        (0, 0, 2, 7),
        (2, 5, 0, 1),
        (2, 5, 3, 7),
        (2, 5, 5, 7),
        (2, 5, 6, 1),
    ],
)
def test_gather_matches_float64_sparse_transpose_reference(
    min_level: int,
    max_level: int,
    target_level: int,
    feature_dim: int,
):
    """Validate gather against an implementation-independent CPU transpose."""
    torch.manual_seed(29)
    grad_target = torch.randn(
        (2, ds.feats_at_level(target_level), feature_dim),
        dtype=torch.float32,
        device="cuda",
    )
    grad_gather = _native_accumulation_backward_gather(
        min_level, max_level, target_level, grad_target
    )
    plan = _multires._get_accumulation_plan(
        min_level, max_level, target_level, grad_target.device
    )
    indices = plan.forward_indices.cpu()
    weights = plan.forward_weights.cpu()
    source_size = sum(
        ds.feats_at_level(level)
        for level in range(min_level, max_level + 1)
    )
    reference = torch.zeros(
        (grad_target.shape[0], source_size, grad_target.shape[2]),
        dtype=torch.float64,
    )
    grad_cpu = grad_target.cpu().to(torch.float64)
    for target in range(indices.shape[0]):
        for slot in range(indices.shape[1]):
            for contribution in range(3):
                weight = float(weights[target, slot, contribution])
                if weight != 0.0:
                    source = int(indices[target, slot, contribution])
                    reference[:, source] += weight * grad_cpu[:, target]
    torch.cuda.synchronize()
    torch.testing.assert_close(
        grad_gather.cpu().to(torch.float64),
        reference,
        rtol=1e-5,
        atol=3e-5,
    )


@pytest.mark.parametrize(("num_triangles", "feature_dim"), [(0, 7), (3, 0)])
def test_accumulation_empty_dimensions(num_triangles: int, feature_dim: int):
    source_size = sum(ds.feats_at_level(level) for level in range(2, 6))
    features = torch.empty(
        (num_triangles, source_size, feature_dim),
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    output = ds.accumulate_to_level(2, 5, features, 5)
    output.sum().backward()
    torch.cuda.synchronize()
    assert output.shape == (
        num_triangles,
        ds.feats_at_level(5),
        feature_dim,
    )
    assert features.grad.shape == features.shape


def test_gather_writes_exact_zeros_for_unused_sources():
    torch.manual_seed(31)
    min_level, max_level, target_level = 2, 5, 0
    grad_target = torch.randn(
        (3, ds.feats_at_level(target_level), 7),
        dtype=torch.float32,
        device="cuda",
    )
    grad_gather = _native_accumulation_backward_gather(
        min_level, max_level, target_level, grad_target
    )
    plan = _multires._get_accumulation_plan(
        min_level, max_level, target_level, grad_target.device
    )
    offsets = plan.reverse_offsets.cpu()
    unused = offsets[1:] == offsets[:-1]
    torch.cuda.synchronize()
    assert unused.any()
    assert torch.count_nonzero(grad_gather[:, unused]).item() == 0


def test_gather_backward_is_bitwise_deterministic():
    torch.manual_seed(37)
    grad_target = torch.randn(
        (17, ds.feats_at_level(5), 7),
        dtype=torch.float32,
        device="cuda",
    )
    first = _native_accumulation_backward_gather(2, 5, 5, grad_target)
    second = _native_accumulation_backward_gather(2, 5, 5, grad_target)
    torch.cuda.synchronize()
    assert torch.equal(first, second)


def test_accumulation_adjoint_identity_and_finite_difference():
    torch.manual_seed(41)
    features = torch.randn(
        (2, ds.feats_at_level(0), 3),
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    output = ds.accumulate_to_level(0, 0, features, 2)
    grad_target = torch.randn_like(output)
    left = (output.to(torch.float64) * grad_target.to(torch.float64)).sum()
    output.backward(grad_target)
    right = (
        features.detach().to(torch.float64)
        * features.grad.detach().to(torch.float64)
    ).sum()
    torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-5)

    index = (0, 1, 2)
    epsilon = 1e-3
    base = features.detach()
    plus = base.clone()
    minus = base.clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    objective_plus = (
        ds.accumulate_to_level(0, 0, plus, 2) * grad_target
    ).sum()
    objective_minus = (
        ds.accumulate_to_level(0, 0, minus, 2) * grad_target
    ).sum()
    numerical = (objective_plus - objective_minus) / (2.0 * epsilon)
    torch.testing.assert_close(
        numerical, features.grad[index], rtol=2e-3, atol=2e-3
    )


def test_gather_accepts_noncontiguous_grad_output():
    torch.manual_seed(43)
    source_size = sum(ds.feats_at_level(level) for level in range(2, 6))
    features = torch.randn(
        (3, source_size, 7), dtype=torch.float32, device="cuda"
    )
    storage = torch.randn(
        (3, ds.feats_at_level(5), 14), dtype=torch.float32, device="cuda"
    )
    grad_target = storage[..., ::2]
    assert not grad_target.is_contiguous()

    expected_features = features.clone().requires_grad_(True)
    ds.accumulate_to_level(2, 5, expected_features, 5).backward(
        grad_target.contiguous()
    )
    actual_features = features.clone().requires_grad_(True)
    ds.accumulate_to_level(2, 5, actual_features, 5).backward(grad_target)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual_features.grad, expected_features.grad, rtol=0, atol=0
    )


def test_cached_gather_plan_is_safe_across_concurrent_streams():
    torch.manual_seed(47)
    cases = [(2, 5, 5, 7), (2, 5, 6, 1)]
    default_stream = torch.cuda.current_stream()
    expected = []
    inputs = []
    grads = []
    for min_level, max_level, target_level, feature_dim in cases:
        source_size = sum(
            ds.feats_at_level(level)
            for level in range(min_level, max_level + 1)
        )
        feature = torch.randn(
            (3, source_size, feature_dim),
            dtype=torch.float32,
            device="cuda",
        )
        grad = torch.randn(
            (3, ds.feats_at_level(target_level), feature_dim),
            dtype=torch.float32,
            device="cuda",
        )
        reference = feature.clone().requires_grad_(True)
        ds.accumulate_to_level(
            min_level, max_level, reference, target_level
        ).backward(grad)
        expected.append(reference.grad)
        inputs.append(feature)
        grads.append(grad)
    default_stream.synchronize()

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    actual = []
    for stream, case, feature, grad in zip(streams, cases, inputs, grads):
        min_level, max_level, target_level, _ = case
        stream.wait_stream(default_stream)
        with torch.cuda.stream(stream):
            candidate = feature.clone().requires_grad_(True)
            ds.accumulate_to_level(
                min_level, max_level, candidate, target_level
            ).backward(grad)
            actual.append(candidate)
    for stream in streams:
        stream.synchronize()
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(
            candidate.grad, reference, rtol=1e-5, atol=3e-5
        )


def _split_edges_all_views_reference(
    resolution: tuple[int, int],
    mvps: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    level: int,
    alpha: torch.Tensor,
    tau_ratio: float,
    num_views_cap: int,
    generator: torch.Generator,
):
    """Run the former batched all-view visibility path."""
    from examples import utils as example_utils

    height, width = resolution
    num_views = mvps.shape[0]
    num_original_faces = faces.shape[0]
    device = vertices.device
    permutation = torch.randperm(
        num_views,
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    selected_mvps = mvps[permutation[: min(num_views, num_views_cap)]]
    clip_vertices = example_utils.project_vertices(vertices, selected_mvps)
    raster = ds.rasterize_multires_triangle_alpha(
        (height, width),
        clip_vertices,
        faces,
        level,
        alpha,
        stochastic=False,
    )

    face_map = torch.arange(
        faces.shape[0], device=device, dtype=torch.long
    )
    vertex_recipes = []
    for view_index in range(selected_mvps.shape[0]):
        raster_view = raster[view_index]
        visible_indices = (
            raster_view[raster_view[..., -1] > 0][..., -1].int() - 1
        ).unique().ravel()
        visible_original = torch.zeros(
            num_original_faces, dtype=torch.int32, device=device
        )
        visible_original[visible_indices] = 1
        visible_faces = visible_original[face_map].contiguous()
        outputs = ds.split_triangle_soup_clip_until(
            (height, width),
            selected_mvps[view_index],
            vertices,
            faces,
            visible_faces,
            tau_ratio=tau_ratio,
            return_vertex_provenance=True,
        )
        next_vertices, next_faces, next_face_map = outputs[:3]
        if next_faces.shape[0] == faces.shape[0]:
            continue
        vertex_recipes.append((outputs[4], outputs[5]))
        vertices, faces = next_vertices, next_faces
        face_map = face_map[next_face_map].contiguous()
    return vertices, faces, face_map, vertex_recipes


def test_per_view_visibility_precompute_matches_all_views_reference():
    from examples import utils as example_utils

    vertices = torch.tensor(
        [
            [-0.85, -0.60, 0.10],
            [-0.25, -0.60, 0.10],
            [-0.55, -0.10, 0.10],
            [0.15, -0.55, 0.20],
            [0.75, -0.55, 0.20],
            [0.45, 0.00, 0.20],
            [-0.25, 0.20, 0.30],
            [0.35, 0.20, 0.30],
            [0.05, 0.75, 0.30],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    faces = torch.arange(
        9, dtype=torch.int32, device="cuda"
    ).reshape(3, 3)
    mvps = torch.eye(
        4, dtype=torch.float32, device="cuda"
    ).repeat(3, 1, 1)
    mvps[0, 0, 3] = 1.05
    mvps[1, 0, 3] = -1.05
    mvps[2, 1, 3] = -0.70
    alpha = torch.full(
        (3, ds.feats_at_level(0), 1),
        0.9,
        dtype=torch.float32,
        device="cuda",
    )
    reference_generator = torch.Generator(device="cuda").manual_seed(53)
    actual_generator = torch.Generator(device="cuda").manual_seed(53)
    arguments = ((48, 64), mvps, vertices, faces, 0, alpha, 0.11, 3)

    expected = _split_edges_all_views_reference(
        *arguments, reference_generator
    )
    actual = example_utils.split_edges_from_training_views(
        *arguments,
        actual_generator,
        return_vertex_provenance=True,
    )
    torch.cuda.synchronize()

    assert expected[1].shape[0] > faces.shape[0]
    assert len(expected[3]) == 3
    for actual_tensor, expected_tensor in zip(actual[:3], expected[:3]):
        assert torch.equal(actual_tensor, expected_tensor)
    assert len(actual[3]) == len(expected[3])
    for actual_recipe, expected_recipe in zip(actual[3], expected[3]):
        assert torch.equal(actual_recipe[0], expected_recipe[0])
        assert torch.equal(actual_recipe[1], expected_recipe[1])
