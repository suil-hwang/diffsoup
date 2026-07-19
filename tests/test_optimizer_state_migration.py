import importlib.util
from pathlib import Path
import sys
from typing import cast

import pytest
import torch


_OPTIMIZE_SPEC = importlib.util.spec_from_file_location(
    "diffsoup_optimize_under_test",
    Path(__file__).parents[1] / "python" / "diffsoup" / "optimize.py",
)
assert _OPTIMIZE_SPEC is not None and _OPTIMIZE_SPEC.loader is not None
optimize = importlib.util.module_from_spec(_OPTIMIZE_SPEC)
sys.modules[_OPTIMIZE_SPEC.name] = optimize
_OPTIMIZE_SPEC.loader.exec_module(optimize)

VectorAdam = optimize.VectorAdam
replace_optimizer_parameter_ = optimize.replace_optimizer_parameter_
replace_vector_adam_parameter_ = optimize.replace_vector_adam_parameter_

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="optimizer-state migration requires CUDA",
    ),
]

DEVICE = torch.device("cuda")
ADAM_OPTIONS = {
    "lr": 0.017,
    "betas": (0.7, 0.91),
    "eps": 3e-6,
    "weight_decay": 0.04,
    "amsgrad": True,
    "maximize": True,
    "fused": True,
}
VECTOR_OPTIONS = {"lr": 0.031, "betas": (0.6, 0.85)}


def _parameter(rows: int, columns: int) -> torch.Tensor:
    values = torch.linspace(
        -0.8,
        1.1,
        rows * columns,
        dtype=torch.float32,
        device=DEVICE,
    )
    return values.reshape(rows, columns).clone().requires_grad_()


def _step(optimizer, parameter: torch.Tensor, gradient: torch.Tensor) -> None:
    parameter.grad = gradient.clone()
    optimizer.step()
    parameter.grad = None


def _state_clone(state: dict) -> dict:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def _group_options(optimizer) -> dict:
    return {
        key: value
        for key, value in optimizer.param_groups[0].items()
        if key != "params"
    }


def _weighted_rows(
    values: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    selected = values.index_select(0, indices.reshape(-1)).reshape(
        *indices.shape,
        *values.shape[1:],
    )
    expanded_weights = weights.reshape(
        *weights.shape,
        *((1,) * (values.ndim - 1)),
    )
    return selected.mul(expanded_weights).sum(dim=1).contiguous()


def _assert_exact_state(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if torch.is_tensor(expected_value):
            assert torch.equal(actual_value, expected_value), key
        else:
            assert actual_value == expected_value, key


def test_fused_adam_prune_duplicate_preserves_state_and_next_step() -> None:
    old_parameter = _parameter(5, 2)
    optimizer = torch.optim.Adam([old_parameter], **ADAM_OPTIONS)
    for index in range(3):
        gradient = (
            torch.arange(10, dtype=torch.float32, device=DEVICE).reshape(5, 2)
            - 4.0
            + 0.37 * index
        )
        _step(optimizer, old_parameter, gradient)

    old_state = _state_clone(optimizer.state[old_parameter])
    old_options = _group_options(optimizer)
    old_defaults = optimizer.defaults.copy()
    parent_map = torch.tensor([4, 1, 1, 3], dtype=torch.int64, device=DEVICE)
    new_parameter = (
        old_parameter.detach().index_select(0, parent_map).clone().requires_grad_()
    )
    replace_optimizer_parameter_(
        optimizer,
        old_parameter,
        new_parameter,
        parent_map,
    )

    expected_state = {
        key: (
            value.index_select(0, parent_map)
            if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
            else value
        )
        for key, value in old_state.items()
    }
    assert optimizer.param_groups[0]["params"][0] is new_parameter
    assert old_parameter not in optimizer.state
    assert _group_options(optimizer) == old_options
    assert optimizer.defaults == old_defaults
    _assert_exact_state(optimizer.state[new_parameter], expected_state)
    assert optimizer.state[new_parameter]["step"].dtype == torch.float32
    assert optimizer.state[new_parameter]["step"].device == new_parameter.device

    reference_parameter = new_parameter.detach().clone().requires_grad_()
    reference = torch.optim.Adam([reference_parameter], **ADAM_OPTIONS)
    reference.state[reference_parameter] = _state_clone(expected_state)
    next_gradient = torch.tensor(
        [[0.3, -0.1], [1.1, -0.7], [-0.4, 0.9], [0.2, -1.3]],
        dtype=torch.float32,
        device=DEVICE,
    )
    _step(optimizer, new_parameter, next_gradient)
    _step(reference, reference_parameter, next_gradient)

    assert torch.equal(new_parameter, reference_parameter)
    _assert_exact_state(
        optimizer.state[new_parameter],
        reference.state[reference_parameter],
    )


def test_lazy_state_and_zero_row_adam_migration() -> None:
    lazy_old = _parameter(3, 2)
    lazy_adam = torch.optim.Adam([lazy_old], **ADAM_OPTIONS)
    parent_map = torch.tensor([2, 0, 0], dtype=torch.int32, device=DEVICE)
    lazy_new = (
        lazy_old.detach().index_select(0, parent_map.long()).clone().requires_grad_()
    )
    replace_optimizer_parameter_(lazy_adam, lazy_old, lazy_new, parent_map)
    assert lazy_new not in lazy_adam.state
    _step(lazy_adam, lazy_new, torch.full_like(lazy_new, 0.25))
    assert lazy_adam.state[lazy_new]["step"].item() == 1

    lazy_vector_old = _parameter(2, 3)
    lazy_vector = VectorAdam([lazy_vector_old], **VECTOR_OPTIONS)
    vector_map = torch.tensor([1, 1, 0], dtype=torch.int64, device=DEVICE)
    lazy_vector_new = (
        lazy_vector_old.detach()
        .index_select(0, vector_map)
        .clone()
        .requires_grad_()
    )
    replace_optimizer_parameter_(
        lazy_vector,
        lazy_vector_old,
        lazy_vector_new,
        vector_map,
    )
    assert lazy_vector_new not in lazy_vector.state
    _step(lazy_vector, lazy_vector_new, torch.full_like(lazy_vector_new, -0.3))
    assert lazy_vector.state[lazy_vector_new]["step"] == 1

    initialized_old = _parameter(3, 2)
    initialized_adam = torch.optim.Adam([initialized_old], **ADAM_OPTIONS)
    _step(
        initialized_adam,
        initialized_old,
        torch.full_like(initialized_old, -0.4),
    )
    old_step = initialized_adam.state[initialized_old]["step"].clone()
    empty_map = torch.empty(0, dtype=torch.int64, device=DEVICE)
    empty_parameter = torch.empty(
        (0, 2),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )
    replace_optimizer_parameter_(
        initialized_adam,
        initialized_old,
        empty_parameter,
        empty_map,
    )
    empty_state = initialized_adam.state[empty_parameter]
    assert empty_state["exp_avg"].shape == (0, 2)
    assert empty_state["exp_avg_sq"].shape == (0, 2)
    assert empty_state["max_exp_avg_sq"].shape == (0, 2)
    assert torch.equal(empty_state["step"], old_step)


def test_basis_lift_resets_only_replaced_adam_state() -> None:
    soup_old = _parameter(3, 2)
    vertices = _parameter(4, 3)
    soup = torch.optim.Adam([soup_old], **ADAM_OPTIONS)
    vertex = VectorAdam([vertices], **VECTOR_OPTIONS)
    _step(soup, soup_old, torch.full_like(soup_old, 0.2))
    _step(vertex, vertices, torch.full_like(vertices, -0.3))

    soup_identity = id(soup)
    vertex_state = _state_clone(vertex.state[vertices])
    soup_new = _parameter(3, 7)
    replace_optimizer_parameter_(soup, soup_old, soup_new, None)

    assert id(soup) == soup_identity
    assert soup.param_groups[0]["params"][0] is soup_new
    assert soup_old not in soup.state and soup_new not in soup.state
    _assert_exact_state(vertex.state[vertices], vertex_state)


def test_basis_lift_preserves_other_adam_group_and_state() -> None:
    feature_old = _parameter(3, 2)
    alpha = _parameter(3, 1)
    optimizer = torch.optim.Adam(
        [
            {"params": [feature_old], "lr": 0.017},
            {"params": [alpha], "lr": 0.023},
        ],
        betas=(0.7, 0.91),
        eps=3e-6,
        fused=True,
    )
    for index in range(3):
        feature_old.grad = torch.full_like(feature_old, 0.2 + 0.1 * index)
        alpha.grad = torch.full_like(alpha, -0.4 + 0.05 * index)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    alpha_state = _state_clone(optimizer.state[alpha])
    group_ids = [id(group) for group in optimizer.param_groups]
    group_options = [
        {
            key: value
            for key, value in group.items()
            if key != "params"
        }
        for group in optimizer.param_groups
    ]
    feature_new = _parameter(3, 7)
    replace_optimizer_parameter_(
        optimizer,
        feature_old,
        feature_new,
        None,
    )

    assert [id(group) for group in optimizer.param_groups] == group_ids
    assert optimizer.param_groups[0]["params"][0] is feature_new
    assert optimizer.param_groups[1]["params"][0] is alpha
    assert feature_old not in optimizer.state
    assert feature_new not in optimizer.state
    _assert_exact_state(optimizer.state[alpha], alpha_state)
    assert [
        {
            key: value
            for key, value in group.items()
            if key != "params"
        }
        for group in optimizer.param_groups
    ] == group_options


def test_vector_adam_identity_copy_preserves_state_and_next_step() -> None:
    old_parameter = _parameter(4, 3)
    optimizer = VectorAdam([old_parameter], **VECTOR_OPTIONS)
    for index in range(3):
        gradient = (
            torch.arange(12, dtype=torch.float32, device=DEVICE).reshape(4, 3)
            / 7.0
            - 0.6 * index
        )
        _step(optimizer, old_parameter, gradient)

    old_state = _state_clone(optimizer.state[old_parameter])
    parent_map = torch.arange(4, dtype=torch.int64, device=DEVICE)
    new_parameter = (
        old_parameter.detach().index_select(0, parent_map).clone().requires_grad_()
    )
    replace_optimizer_parameter_(
        optimizer,
        old_parameter,
        new_parameter,
        parent_map,
    )
    expected_state = {
        "step": old_state["step"],
        "g1": old_state["g1"].index_select(0, parent_map),
        "g2": old_state["g2"].index_select(0, parent_map),
    }
    _assert_exact_state(optimizer.state[new_parameter], expected_state)

    reference_parameter = new_parameter.detach().clone().requires_grad_()
    reference = VectorAdam([reference_parameter], **VECTOR_OPTIONS)
    reference.state[reference_parameter] = _state_clone(expected_state)
    next_gradient = torch.tensor(
        [
            [0.1, -0.2, 0.3],
            [-0.5, 0.7, -0.9],
            [1.2, -0.4, 0.8],
            [-0.3, -0.6, 1.1],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    _step(optimizer, new_parameter, next_gradient)
    _step(reference, reference_parameter, next_gradient)

    assert torch.equal(new_parameter, reference_parameter)
    _assert_exact_state(
        optimizer.state[new_parameter],
        reference.state[reference_parameter],
    )


def test_initialized_vector_adam_subset_and_zero_row_migration() -> None:
    old_parameter = _parameter(5, 3)
    optimizer = VectorAdam([old_parameter], **VECTOR_OPTIONS)
    for index in range(3):
        gradient = (
            torch.arange(15, dtype=torch.float32, device=DEVICE).reshape(5, 3)
            / 9.0
            - 0.4 * index
        )
        _step(optimizer, old_parameter, gradient)

    old_state = _state_clone(optimizer.state[old_parameter])
    parent_map = torch.tensor([4, 1, 1], dtype=torch.int32, device=DEVICE)
    new_parameter = (
        old_parameter.detach()
        .index_select(0, parent_map.long())
        .clone()
        .requires_grad_()
    )
    replace_optimizer_parameter_(
        optimizer,
        old_parameter,
        new_parameter,
        parent_map,
    )
    expected_state = {
        "step": old_state["step"],
        "g1": old_state["g1"].index_select(0, parent_map.long()),
        "g2": old_state["g2"].index_select(0, parent_map.long()),
    }
    _assert_exact_state(optimizer.state[new_parameter], expected_state)

    reference_parameter = new_parameter.detach().clone().requires_grad_()
    reference = VectorAdam([reference_parameter], **VECTOR_OPTIONS)
    reference.state[reference_parameter] = _state_clone(expected_state)
    next_gradient = torch.tensor(
        [
            [0.2, -0.1, 0.4],
            [-0.7, 0.8, -0.3],
            [0.6, -0.5, 0.9],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    _step(optimizer, new_parameter, next_gradient)
    _step(reference, reference_parameter, next_gradient)
    assert torch.equal(new_parameter, reference_parameter)
    _assert_exact_state(
        optimizer.state[new_parameter],
        reference.state[reference_parameter],
    )

    state_before_empty = _state_clone(optimizer.state[new_parameter])
    empty_map = torch.empty(0, dtype=torch.int64, device=DEVICE)
    empty_parameter = torch.empty(
        (0, 3),
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )
    replace_optimizer_parameter_(
        optimizer,
        new_parameter,
        empty_parameter,
        empty_map,
    )
    expected_empty_state = {
        "step": state_before_empty["step"],
        "g1": state_before_empty["g1"].index_select(0, empty_map),
        "g2": state_before_empty["g2"].index_select(0, empty_map),
    }
    assert optimizer.param_groups[0]["params"][0] is empty_parameter
    assert new_parameter not in optimizer.state
    _assert_exact_state(optimizer.state[empty_parameter], expected_empty_state)


def test_vector_adam_direct_affine_recipe_preserves_next_step() -> None:
    old_parameter = _parameter(4, 3)
    optimizer = VectorAdam([old_parameter], **VECTOR_OPTIONS)
    for index in range(4):
        gradient = torch.tensor(
            [
                [0.2, -0.7, 1.1],
                [-0.4, 0.9, 0.3],
                [1.3, -0.2, -0.8],
                [0.6, 0.5, -1.0],
            ],
            dtype=torch.float32,
            device=DEVICE,
        ) + 0.19 * index
        _step(optimizer, old_parameter, gradient)

    indices = torch.tensor(
        [[0, 1, 0], [2, 3, 1], [1, 3, 1]],
        dtype=torch.int32,
        device=DEVICE,
    )
    weights = torch.tensor(
        [[0.25, 0.75, 0.0], [0.2, 0.3, 0.5], [0.6, 0.4, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    new_parameter = (
        _weighted_rows(old_parameter.detach(), indices.long(), weights)
        .clone()
        .requires_grad_()
    )
    old_state = optimizer.state[old_parameter]
    expected_state = {
        "step": old_state["step"],
        "g1": _weighted_rows(old_state["g1"], indices.long(), weights),
        "g2": _weighted_rows(old_state["g2"], indices.long(), weights),
    }
    replace_vector_adam_parameter_(
        optimizer,
        old_parameter,
        new_parameter,
        [(indices, weights)],
    )
    _assert_exact_state(optimizer.state[new_parameter], expected_state)

    reference_parameter = new_parameter.detach().clone().requires_grad_()
    reference = VectorAdam([reference_parameter], **VECTOR_OPTIONS)
    reference.state[reference_parameter] = _state_clone(expected_state)
    next_gradient = torch.tensor(
        [[0.4, -0.1, 0.8], [-0.6, 1.0, 0.2], [0.9, -0.5, -0.7]],
        dtype=torch.float32,
        device=DEVICE,
    )
    _step(optimizer, new_parameter, next_gradient)
    _step(reference, reference_parameter, next_gradient)

    assert torch.equal(new_parameter, reference_parameter)
    _assert_exact_state(
        optimizer.state[new_parameter],
        reference.state[reference_parameter],
    )


def test_vector_adam_sequential_recipes_preserve_next_step() -> None:
    old_parameter = _parameter(4, 3)
    optimizer = VectorAdam([old_parameter], **VECTOR_OPTIONS)
    for index in range(3):
        gradient = torch.tensor(
            [
                [0.2, -0.7, 1.1],
                [-0.4, 0.9, 0.3],
                [1.3, -0.2, -0.8],
                [0.6, 0.5, -1.0],
            ],
            dtype=torch.float32,
            device=DEVICE,
        ) + 0.13 * index
        _step(optimizer, old_parameter, gradient)

    indices_1 = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [0, 2, 0],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    weights_1 = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.35, 0.65, 0.0],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    indices_2 = torch.tensor(
        [[0, 4, 0], [1, 3, 4], [2, 4, 2]],
        dtype=torch.int64,
        device=DEVICE,
    )
    weights_2 = torch.tensor(
        [[0.2, 0.8, 0.0], [0.25, 0.5, 0.25], [0.6, 0.4, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    intermediate = _weighted_rows(
        old_parameter.detach(),
        indices_1.long(),
        weights_1,
    )
    new_parameter = (
        _weighted_rows(intermediate, indices_2, weights_2)
        .clone()
        .requires_grad_()
    )
    old_state = optimizer.state[old_parameter]
    expected_state = {
        "step": old_state["step"],
        "g1": _weighted_rows(
            _weighted_rows(old_state["g1"], indices_1.long(), weights_1),
            indices_2,
            weights_2,
        ),
        "g2": _weighted_rows(
            _weighted_rows(old_state["g2"], indices_1.long(), weights_1),
            indices_2,
            weights_2,
        ),
    }
    replace_vector_adam_parameter_(
        optimizer,
        old_parameter,
        new_parameter,
        [(indices_1, weights_1), (indices_2, weights_2)],
    )
    _assert_exact_state(optimizer.state[new_parameter], expected_state)

    reference_parameter = new_parameter.detach().clone().requires_grad_()
    reference = VectorAdam([reference_parameter], **VECTOR_OPTIONS)
    reference.state[reference_parameter] = _state_clone(expected_state)
    next_gradient = torch.tensor(
        [[0.4, -0.1, 0.8], [-0.6, 1.0, 0.2], [0.9, -0.5, -0.7]],
        dtype=torch.float32,
        device=DEVICE,
    )
    _step(optimizer, new_parameter, next_gradient)
    _step(reference, reference_parameter, next_gradient)
    assert torch.equal(new_parameter, reference_parameter)
    _assert_exact_state(
        optimizer.state[new_parameter],
        reference.state[reference_parameter],
    )


def test_linear_g2_transport_equals_affine_g2_plus_pairwise_dispersion() -> None:
    torch.manual_seed(23)
    history = torch.randn(19, 3, 3, dtype=torch.float64, device=DEVICE)
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64, device=DEVICE)
    beta2 = 0.85
    ema_weights = (1.0 - beta2) * beta2 ** torch.arange(
        history.shape[0] - 1,
        -1,
        -1,
        dtype=torch.float64,
        device=DEVICE,
    )

    source_g2 = (
        ema_weights[:, None]
        * history.square().sum(dim=-1)
    ).sum(dim=0)
    linear_g2 = torch.dot(weights, source_g2)
    affine_history = (history * weights[None, :, None]).sum(dim=1)
    affine_g2 = torch.dot(
        ema_weights,
        affine_history.square().sum(dim=-1),
    )
    pairwise_distance = (
        history[:, :, None, :] - history[:, None, :, :]
    ).square().sum(dim=-1)
    dispersion = 0.5 * (
        ema_weights[:, None, None]
        * weights[None, :, None]
        * weights[None, None, :]
        * pairwise_distance
    ).sum()

    torch.testing.assert_close(
        linear_g2,
        affine_g2 + dispersion,
        rtol=1e-12,
        atol=1e-12,
    )
    assert dispersion.item() > 0

    identical = history[:, :1].expand(-1, 3, -1)
    identical_source_g2 = (
        ema_weights[:, None]
        * identical.square().sum(dim=-1)
    ).sum(dim=0)
    identical_linear = torch.dot(weights, identical_source_g2)
    identical_affine = torch.dot(
        ema_weights,
        (identical * weights[None, :, None]).sum(dim=1).square().sum(dim=-1),
    )
    torch.testing.assert_close(
        identical_linear,
        identical_affine,
        rtol=1e-12,
        atol=1e-12,
    )


def test_native_midpoint_g2_transport_is_a_convex_warm_start() -> None:
    import diffsoup as ds

    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)
    outputs = cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        ds.split_triangle_soup(
            vertices,
            faces,
            num_splits=1,
            return_vertex_provenance=True,
        ),
    )
    new_vertices, _, _, _, source_indices, source_weights = outputs
    new_vertices.requires_grad_(True)

    source_gradients = torch.tensor(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    source_g2 = source_gradients.square().sum(dim=-1, keepdim=True)
    optimizer = VectorAdam([vertices], **VECTOR_OPTIONS)
    optimizer.state[vertices] = {
        "step": 1,
        "g1": source_gradients.clone(),
        "g2": source_g2.clone(),
    }

    replace_vector_adam_parameter_(
        optimizer,
        vertices,
        new_vertices,
        [(source_indices, source_weights)],
    )
    migrated_g2 = optimizer.state[new_vertices]["g2"]
    expected_linear = _weighted_rows(
        source_g2,
        source_indices.long(),
        source_weights,
    )
    torch.testing.assert_close(migrated_g2, expected_linear, rtol=0, atol=0)

    midpoint = (
        (source_indices[:, 0] != source_indices[:, 1])
        & torch.isclose(source_weights[:, 0], torch.tensor(0.5, device=DEVICE))
        & torch.isclose(source_weights[:, 1], torch.tensor(0.5, device=DEVICE))
        & (source_weights[:, 2] == 0)
    )
    assert midpoint.sum().item() == 2
    torch.testing.assert_close(
        migrated_g2[midpoint],
        torch.full((2, 1), 17.0, device=DEVICE),
        rtol=0,
        atol=0,
    )

    affine_gradient = _weighted_rows(
        source_gradients,
        source_indices.long(),
        source_weights,
    )
    affine_gradient_g2 = affine_gradient.square().sum(dim=-1, keepdim=True)
    torch.testing.assert_close(
        affine_gradient_g2[midpoint],
        torch.full((2, 1), 16.0, device=DEVICE),
        rtol=0,
        atol=0,
    )
    assert not torch.equal(migrated_g2[midpoint], affine_gradient_g2[midpoint])

    constant_g2 = torch.full((2,), 8.0, device=DEVICE)
    midpoint_weights = torch.tensor([0.5, 0.5], device=DEVICE)
    assert torch.dot(midpoint_weights, constant_g2).item() == 8.0
    assert torch.dot(midpoint_weights.square(), constant_g2).item() == 4.0


def test_split_face_feature_alpha_values_and_moments_copy_parent_rows() -> None:
    import diffsoup as ds

    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.5, 0.0, 0.0],
            [10.0, 0.5, 0.0],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5]],
        dtype=torch.int32,
        device=DEVICE,
    )
    _, _, face_map, _ = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ds.split_triangle_soup(
            vertices,
            faces,
            num_splits=1,
        ),
    )
    parent_map = face_map.long().contiguous()
    assert torch.equal(
        parent_map,
        torch.tensor([0, 1, 0], device=DEVICE),
    )

    feature = torch.linspace(
        -0.7,
        1.1,
        2 * 3 * 7,
        device=DEVICE,
    ).reshape(2, 3, 7).clone().requires_grad_()
    alpha = torch.linspace(
        -1.3,
        0.9,
        2 * 3,
        device=DEVICE,
    ).reshape(2, 3, 1).clone().requires_grad_()
    optimizer = torch.optim.Adam(
        [
            {"params": [feature], "lr": 0.017},
            {"params": [alpha], "lr": 0.023},
        ],
        betas=(0.7, 0.91),
        eps=3e-6,
        amsgrad=True,
        fused=True,
    )
    feature.grad = torch.linspace(
        -0.4,
        0.8,
        feature.numel(),
        device=DEVICE,
    ).reshape_as(feature)
    alpha.grad = torch.linspace(
        0.6,
        -0.2,
        alpha.numel(),
        device=DEVICE,
    ).reshape_as(alpha)
    optimizer.step()
    feature.grad = alpha.grad = None

    feature_value = feature.detach().clone()
    alpha_value = alpha.detach().clone()
    feature_state = _state_clone(optimizer.state[feature])
    alpha_state = _state_clone(optimizer.state[alpha])
    with torch.no_grad():
        new_feature = ds.expand_by_index(feature, parent_map)
        new_alpha = ds.expand_by_index(alpha, parent_map)
    new_feature.requires_grad_(True)
    new_alpha.requires_grad_(True)
    replace_optimizer_parameter_(
        optimizer,
        feature,
        new_feature,
        parent_map,
    )
    replace_optimizer_parameter_(
        optimizer,
        alpha,
        new_alpha,
        parent_map,
    )

    assert torch.equal(new_feature, feature_value.index_select(0, parent_map))
    assert torch.equal(new_alpha, alpha_value.index_select(0, parent_map))
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        assert torch.equal(
            optimizer.state[new_feature][key],
            feature_state[key].index_select(0, parent_map),
        )
        assert torch.equal(
            optimizer.state[new_alpha][key],
            alpha_state[key].index_select(0, parent_map),
        )
    assert torch.equal(
        optimizer.state[new_feature]["step"],
        feature_state["step"],
    )
    assert torch.equal(
        optimizer.state[new_alpha]["step"],
        alpha_state["step"],
    )

    new_feature.grad = torch.zeros_like(new_feature)
    new_alpha.grad = torch.zeros_like(new_alpha)
    new_feature.grad[0].fill_(0.4)
    new_feature.grad[2].fill_(-0.6)
    new_alpha.grad[0].fill_(0.7)
    new_alpha.grad[2].fill_(-0.3)
    optimizer.step()
    assert not torch.equal(new_feature[0], new_feature[2])
    assert not torch.equal(new_alpha[0], new_alpha[2])


def test_parent_row_copy_is_not_child_local_field_reparameterization() -> None:
    import diffsoup as ds

    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.5, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)
    outputs = cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        ds.split_triangle_soup(
            vertices,
            faces,
            num_splits=1,
            return_vertex_provenance=True,
        ),
    )
    _, output_faces, face_map, _, source_indices, source_weights = outputs
    alpha_low = torch.sigmoid(torch.tensor(-2.0, device=DEVICE))
    alpha_high = torch.sigmoid(torch.tensor(2.0, device=DEVICE))
    parent_row = torch.tensor(
        [[[0.5, 0.0], [0.1, 0.0], [0.9, 0.0]]],
        dtype=torch.float32,
        device=DEVICE,
    )
    parent_row[0, :, 1] = torch.stack(
        [torch.tensor(0.5, device=DEVICE), alpha_low, alpha_high]
    )
    copied_rows = ds.expand_by_index(parent_row, face_map.long())

    def parent_barycentric(vertex_index: int) -> torch.Tensor:
        barycentric = torch.zeros(3, dtype=torch.float32, device=DEVICE)
        for source_index, source_weight in zip(
            source_indices[vertex_index],
            source_weights[vertex_index],
            strict=True,
        ):
            local = torch.nonzero(
                faces[0] == source_index,
                as_tuple=False,
            ).flatten()
            assert local.numel() == 1
            barycentric[local.item()] += source_weight
        return barycentric

    def evaluate(
        rows: torch.Tensor,
        barycentric: torch.Tensor,
        triangle_ids: torch.Tensor,
    ) -> torch.Tensor:
        raster = torch.zeros(
            (1, 1, barycentric.shape[0], 4),
            dtype=torch.float32,
            device=DEVICE,
        )
        raster[0, 0, :, :2] = barycentric[:, :2]
        raster[0, 0, :, 3] = triangle_ids.to(torch.float32) + 1
        return ds.multires_triangle_color(raster, 0, rows)[0, 0]

    child_vertex_parent_barycentric = torch.stack(
        [
            torch.stack(
                [parent_barycentric(int(vertex)) for vertex in face],
            )
            for face in output_faces
        ]
    )
    child_midpoint_barycentric = torch.zeros(
        (output_faces.shape[0], 3),
        dtype=torch.float32,
        device=DEVICE,
    )
    parent_midpoint_barycentric = []
    for child_index, face in enumerate(output_faces):
        midpoint_local = []
        for local_index, vertex in enumerate(face):
            weights = source_weights[int(vertex)]
            if (
                torch.isclose(weights[0], torch.tensor(0.5, device=DEVICE))
                and torch.isclose(weights[1], torch.tensor(0.5, device=DEVICE))
                and weights[2] == 0
            ):
                midpoint_local.append(local_index)
        assert len(midpoint_local) == 1
        local_index = midpoint_local[0]
        child_midpoint_barycentric[child_index, local_index] = 1.0
        parent_midpoint_barycentric.append(
            child_vertex_parent_barycentric[child_index, local_index]
        )
    parent_midpoint_barycentric = torch.stack(parent_midpoint_barycentric)
    child_ids = torch.arange(output_faces.shape[0], device=DEVICE)
    parent_values = evaluate(
        parent_row,
        parent_midpoint_barycentric,
        torch.zeros_like(child_ids),
    )
    copied_values = evaluate(
        copied_rows,
        child_midpoint_barycentric,
        child_ids,
    )
    torch.testing.assert_close(
        parent_values,
        torch.full_like(parent_values, 0.5),
        rtol=0,
        atol=6e-8,
    )
    torch.testing.assert_close(
        copied_values[:, 0].sort().values,
        torch.tensor([0.1, 0.9], device=DEVICE),
        rtol=0,
        atol=6e-8,
    )
    torch.testing.assert_close(
        copied_values[:, 1].sort().values,
        torch.stack([alpha_low, alpha_high]),
        rtol=0,
        atol=6e-8,
    )

    level_zero_vertex_order = torch.tensor([2, 0, 1], device=DEVICE)
    sample_barycentric = child_vertex_parent_barycentric.index_select(
        1,
        level_zero_vertex_order,
    ).reshape(-1, 3)
    resampled_rows = evaluate(
        parent_row,
        sample_barycentric,
        torch.zeros(sample_barycentric.shape[0], dtype=torch.long, device=DEVICE),
    ).reshape(output_faces.shape[0], 3, 2)
    resampled_values = evaluate(
        resampled_rows,
        child_midpoint_barycentric,
        child_ids,
    )
    torch.testing.assert_close(
        resampled_values,
        parent_values,
        rtol=0,
        atol=6e-8,
    )

    constant_row = torch.full((1, 3, 2), 0.37, device=DEVICE)
    constant_children = ds.expand_by_index(constant_row, face_map.long())
    constant_values = evaluate(
        constant_children,
        child_midpoint_barycentric,
        child_ids,
    )
    torch.testing.assert_close(
        constant_values,
        torch.full_like(constant_values, 0.37),
        rtol=0,
        atol=0,
    )


def test_optimizer_migration_rejects_invalid_recipes() -> None:
    adam_old = _parameter(3, 2)
    adam_new = _parameter(2, 2)
    adam = torch.optim.Adam([adam_old], **ADAM_OPTIONS)
    _step(adam, adam_old, torch.full_like(adam_old, 0.2))
    adam_state = _state_clone(adam.state[adam_old])

    invalid_parent_maps = (
        torch.tensor([0], dtype=torch.int64, device=DEVICE),
        torch.tensor([0, 3], dtype=torch.int64, device=DEVICE),
        torch.tensor([0, -1], dtype=torch.int64, device=DEVICE),
        torch.tensor([[0, 1]], dtype=torch.int64, device=DEVICE),
        torch.tensor([0.0, 1.0], dtype=torch.float32, device=DEVICE),
    )
    for parent_map in invalid_parent_maps:
        with pytest.raises(AssertionError):
            replace_optimizer_parameter_(adam, adam_old, adam_new, parent_map)
        assert adam.param_groups[0]["params"][0] is adam_old
        _assert_exact_state(adam.state[adam_old], adam_state)

    vector_old = _parameter(3, 3)
    vector_new = _parameter(2, 3)
    vector = VectorAdam([vector_old], **VECTOR_OPTIONS)
    _step(vector, vector_old, torch.full_like(vector_old, -0.4))
    vector_state = _state_clone(vector.state[vector_old])
    valid_indices = torch.tensor(
        [[0, 1, 0], [1, 2, 1]], dtype=torch.int64, device=DEVICE
    )
    valid_weights = torch.tensor(
        [[0.25, 0.75, 0.0], [0.2, 0.3, 0.5]],
        dtype=torch.float32,
        device=DEVICE,
    )
    invalid_recipes = (
        (
            torch.tensor(
                [[0, 1, 3], [1, 2, 1]], dtype=torch.int64, device=DEVICE
            ),
            valid_weights,
        ),
        (
            torch.tensor(
                [[0, -1, 0], [1, 2, 1]], dtype=torch.int64, device=DEVICE
            ),
            valid_weights,
        ),
        (
            valid_indices,
            torch.tensor(
                [[0.2, 0.7, 0.0], [0.2, 0.3, 0.5]],
                dtype=torch.float32,
                device=DEVICE,
            ),
        ),
        (
            valid_indices,
            torch.tensor(
                [[1.1, -0.1, 0.0], [0.2, 0.3, 0.5]],
                dtype=torch.float32,
                device=DEVICE,
            ),
        ),
        (
            valid_indices,
            torch.tensor(
                [[float("nan"), 0.0, 1.0], [0.2, 0.3, 0.5]],
                dtype=torch.float32,
                device=DEVICE,
            ),
        ),
        (
            torch.tensor(
                [[0, 1], [1, 2]], dtype=torch.int64, device=DEVICE
            ),
            torch.tensor(
                [[0.25, 0.75], [0.4, 0.6]],
                dtype=torch.float32,
                device=DEVICE,
            ),
        ),
        (valid_indices.to(dtype=torch.float32), valid_weights),
        (valid_indices, valid_weights.to(dtype=torch.float64)),
    )
    for recipe in invalid_recipes:
        with pytest.raises(AssertionError):
            replace_vector_adam_parameter_(
                vector, vector_old, vector_new, [recipe]
            )
        assert vector.param_groups[0]["params"][0] is vector_old
        _assert_exact_state(vector.state[vector_old], vector_state)


def test_world_split_provenance_preserves_legacy_and_reconstructs_vertices() -> None:
    import diffsoup as ds

    vertices = torch.tensor(
        [
            [-1.0, -0.5, 0.2],
            [2.0, -0.25, 0.6],
            [0.1, 1.5, -0.4],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)

    legacy = ds.split_triangle_soup(vertices, faces, num_splits=8)
    outputs = ds.split_triangle_soup(
        vertices,
        faces,
        num_splits=8,
        return_vertex_provenance=True,
    )
    for old, new in zip(legacy, outputs[:4], strict=True):
        assert torch.equal(old, new)

    outputs = cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        outputs,
    )
    output_vertices, _, _, _, source_indices, source_weights = outputs
    reconstructed = (
        vertices[source_indices.long()] * source_weights.unsqueeze(-1)
    ).sum(dim=1)
    torch.testing.assert_close(reconstructed, output_vertices, rtol=0, atol=1e-6)
    torch.testing.assert_close(
        source_weights.sum(dim=1),
        torch.ones(output_vertices.shape[0], device=DEVICE),
        rtol=0,
        atol=1e-6,
    )
    assert source_indices.dtype == torch.int32
    assert source_weights.dtype == torch.float32
    assert source_indices.is_contiguous() and source_weights.is_contiguous()
    assert torch.all(source_indices >= 0)
    assert torch.all(source_indices < vertices.shape[0])
    assert torch.isfinite(source_weights).all()
    assert (source_weights >= 0).all()

    no_op = ds.split_triangle_soup(
        vertices,
        faces,
        num_splits=0,
        return_vertex_provenance=True,
    )
    identity = torch.arange(vertices.shape[0], device=DEVICE, dtype=torch.int32)
    assert torch.equal(no_op[4], identity[:, None].expand(-1, 3))
    expected_weights = torch.zeros_like(no_op[5])
    expected_weights[:, 0] = 1.0
    assert torch.equal(no_op[5], expected_weights)


def test_clip_split_provenance_reconstructs_perspective_vertices() -> None:
    import diffsoup as ds

    vertices = torch.tensor(
        [[-0.5, 0.0, 1.0], [2.0, 0.0, 4.0], [0.0, 0.4, 2.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)
    valid = torch.ones(1, dtype=torch.int32, device=DEVICE)
    mvp = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    args = ((64, 64), mvp, vertices, faces, valid, 4)
    legacy = ds.split_triangle_soup_clip(*args)
    outputs = ds.split_triangle_soup_clip(
        *args,
        return_vertex_provenance=True,
    )
    for old, new in zip(legacy, outputs[:4], strict=True):
        assert torch.equal(old, new)

    outputs = cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        outputs,
    )
    output_vertices, _, _, _, source_indices, source_weights = outputs
    reconstructed = (
        vertices[source_indices.long()] * source_weights.unsqueeze(-1)
    ).sum(dim=1)
    torch.testing.assert_close(reconstructed, output_vertices, rtol=0, atol=5e-7)
    torch.testing.assert_close(
        source_weights.sum(dim=1),
        torch.ones(output_vertices.shape[0], device=DEVICE),
    )
    assert torch.isfinite(source_weights).all()
    assert (source_weights >= 0).all()
    assert torch.equal(
        source_indices[3],
        torch.tensor([0, 1, 0], dtype=torch.int32, device=DEVICE),
    )
    torch.testing.assert_close(
        source_weights[3],
        torch.tensor([0.8, 0.2, 0.0], device=DEVICE),
    )
