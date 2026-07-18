# python/diffsoup/optimize.py
"""Custom optimisers for triangle-soup parameters."""

from collections.abc import Sequence

import torch


class VectorAdam(torch.optim.Optimizer):
    """Adam variant with isotropic second-moment estimation.

    For parameters whose last dimension represents a spatial vector (e.g. XYZ
    positions), the squared-gradient statistics are pooled across that
    dimension so that all components share a single adaptive learning rate.
    This prevents axis-aligned bias in the updates.

    Based on the VectorAdam optimiser introduced in:
        Ling, S. Z., Sharp, N., and Jacobson, A., "VectorAdam for Rotation
        Equivariant Geometry Optimization," NeurIPS 2022.

    Args:
        params: Iterable of parameters to optimise.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for running averages of gradient and its
            squared norm (default: (0.9, 0.999)).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Lazy state initialisation
                if len(state) == 0:
                    state["step"] = 0
                    state["g1"] = torch.zeros_like(p.data)
                    state["g2"] = torch.zeros_like(p.data[..., :1])

                state["step"] += 1
                g1 = state["g1"]
                g2 = state["g2"]

                # Exponential moving averages
                g1.mul_(b1).add_(grad, alpha=1 - b1)
                g2.mul_(b2).add_(
                    grad.square().sum(dim=-1, keepdim=True), alpha=1 - b2
                )

                # Bias correction
                step = state["step"]
                m1 = g1 / (1 - b1**step)
                m2 = g2 / (1 - b2**step)

                # Isotropic normalisation
                p.data.sub_(m1 / (m2.sqrt() + 1e-8), alpha=lr)


@torch.no_grad()
def replace_optimizer_parameter_(
    optimizer: torch.optim.Optimizer,
    old_parameter: torch.Tensor,
    new_parameter: torch.Tensor,
    parent_map: torch.Tensor | None,
) -> None:
    """Replace a parameter, gathering row moments or resetting them on lift."""
    assert old_parameter is not new_parameter and isinstance(
        optimizer, (torch.optim.Adam, VectorAdam)
    )
    group, index = next(
        (group, index)
        for group in optimizer.param_groups
        for index, parameter in enumerate(group["params"])
        if parameter is old_parameter
    )

    if parent_map is None:
        optimizer.state.pop(old_parameter, None)
    else:
        assert (
            parent_map.ndim == 1
            and parent_map.dtype in (torch.int32, torch.int64)
            and parent_map.device == old_parameter.device == new_parameter.device
            and old_parameter.dtype == new_parameter.dtype
            and old_parameter.ndim == new_parameter.ndim > 0
            and old_parameter.shape[1:] == new_parameter.shape[1:]
            and parent_map.shape[0] == new_parameter.shape[0]
            and (
                not parent_map.numel()
                or bool(
                    (
                        (parent_map >= 0)
                        & (parent_map < old_parameter.shape[0])
                    ).all().item()
                )
            )
        )
        state = optimizer.state.pop(old_parameter, None)
        if state:
            keys = (
                ("g1", "g2")
                if isinstance(optimizer, VectorAdam)
                else ("exp_avg", "exp_avg_sq")
            )
            if group.get("amsgrad", False):
                keys += ("max_exp_avg_sq",)
            indices = parent_map.to(dtype=torch.long)
            for key in keys:
                state[key] = state[key].index_select(0, indices).contiguous()
            optimizer.state[new_parameter] = state

    group["params"][index] = new_parameter


@torch.no_grad()
def replace_vector_adam_parameter_(
    optimizer: VectorAdam,
    old_parameter: torch.Tensor,
    new_parameter: torch.Tensor,
    recipes: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Replace vertices and affine-interpolate optimizer moments as a warm start."""
    assert (
        old_parameter is not new_parameter
        and isinstance(optimizer, VectorAdam)
        and old_parameter.ndim >= 2
    )
    group, index = next(
        (group, index)
        for group in optimizer.param_groups
        for index, parameter in enumerate(group["params"])
        if parameter is old_parameter
    )

    row_count = old_parameter.shape[0]
    recipe_tensors = []
    for source_indices, source_weights in recipes:
        assert (
            source_indices.ndim == 2
            and source_indices.shape[1] == 3
            and source_indices.dtype in (torch.int32, torch.int64)
            and source_indices.device == old_parameter.device
            and source_weights.shape == source_indices.shape
            and source_weights.dtype == old_parameter.dtype
            and source_weights.device == old_parameter.device
        )
        invalid = (
            ((source_indices < 0) | (source_indices >= row_count)).any()
            | (source_weights < 0).any()
            | (
                ~torch.isclose(
                    source_weights.sum(dim=1),
                    torch.ones_like(source_weights[:, 0]),
                    rtol=1e-5,
                    atol=1e-6,
                )
            ).any()
        )
        assert not bool(invalid.item())
        recipe_tensors.append((source_indices.to(dtype=torch.long), source_weights))
        row_count = source_indices.shape[0]
    assert (
        old_parameter.ndim == new_parameter.ndim
        and old_parameter.shape[1:] == new_parameter.shape[1:]
        and old_parameter.device == new_parameter.device
        and old_parameter.dtype == new_parameter.dtype
        and row_count == new_parameter.shape[0]
    )

    def transport(value: torch.Tensor) -> torch.Tensor:
        for source_indices, source_weights in recipe_tensors:
            weights = source_weights[(...,) + (None,) * (value.ndim - 1)]
            value = (value[source_indices] * weights).sum(dim=1).contiguous()
        return value

    state = optimizer.state.pop(old_parameter, None)
    if state:
        for key in ("g1", "g2"):
            state[key] = transport(state[key])
        optimizer.state[new_parameter] = state
    group["params"][index] = new_parameter
