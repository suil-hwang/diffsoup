"""Regression tests for ColorMLP's memory-saving split-input path."""

from __future__ import annotations

import copy

import pytest
import torch


ds = pytest.importorskip("diffsoup")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="ColorMLP split-input tests require a CUDA device",
    ),
]


def _legacy_forward(
    model: torch.nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the former full-concatenation path as an independent reference."""
    batch, height, width, _ = features.shape
    rgb = features[..., :3]
    resolution_weight = features[..., 3:4]
    flat = features.view(-1, model.input_dim)
    flat_mask = mask.view(-1)
    output = torch.zeros(
        batch * height * width,
        model.output_dim,
        device=features.device,
        dtype=features.dtype,
    )
    if flat_mask.any():
        valid_output = model.mlp(flat[flat_mask])
        valid_output = (
            (1.0 - resolution_weight[mask]) * rgb[mask]
            + resolution_weight[mask] * valid_output
        )
        output[flat_mask] = valid_output
    return output.view(batch, height, width, model.output_dim)


def test_split_extra_features_matches_legacy_forward_and_backward() -> None:
    """The split path may differ by rounding but must preserve gradients."""
    torch.manual_seed(123)
    batch, height, width = 2, 47, 61
    base = torch.rand(batch, height, width, 7, device="cuda")
    extra = torch.rand(batch, height, width, 9, device="cuda")
    mask = torch.rand(batch, height, width, device="cuda") > 0.45
    grad_output = torch.randn(batch, height, width, 3, device="cuda")

    legacy_model = ds.ColorMLP(16, 3, hidden_dim=16, n_layers=2).cuda()
    split_model = copy.deepcopy(legacy_model)
    legacy_base = base.clone().requires_grad_(True)
    legacy_extra = extra.clone().requires_grad_(True)
    split_base = base.clone().requires_grad_(True)
    split_extra = extra.clone().requires_grad_(True)

    legacy = _legacy_forward(
        legacy_model,
        torch.cat([legacy_base, legacy_extra], dim=-1),
        mask,
    )
    split = split_model(split_base, mask=mask, extra_features=split_extra)
    legacy.backward(grad_output)
    split.backward(grad_output)
    torch.cuda.synchronize()

    torch.testing.assert_close(legacy, split, rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(
        legacy_base.grad,
        split_base.grad,
        rtol=1e-5,
        atol=5e-7,
    )
    assert torch.equal(legacy_extra.grad, split_extra.grad)
    for legacy_parameter, split_parameter in zip(
        legacy_model.parameters(), split_model.parameters(), strict=True
    ):
        assert torch.equal(legacy_parameter.grad, split_parameter.grad)


def test_empty_mask_returns_connected_zero_and_zero_gradients() -> None:
    torch.manual_seed(127)
    model = ds.ColorMLP(16, 3, hidden_dim=16, n_layers=2).cuda()
    base = torch.rand(2, 5, 7, 7, device="cuda", requires_grad=True)
    extra = torch.rand(2, 5, 7, 9, device="cuda", requires_grad=True)
    mask = torch.zeros(2, 5, 7, dtype=torch.bool, device="cuda")

    output = model(base, mask=mask, extra_features=extra)
    assert output.shape == (2, 5, 7, 3)
    assert output.requires_grad
    assert torch.count_nonzero(output).item() == 0

    output.sum().backward()
    assert torch.count_nonzero(base.grad).item() == 0
    assert torch.count_nonzero(extra.grad).item() == 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() == 0


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "mask"])
def test_split_extra_features_validates_layout(invalid: str) -> None:
    model = ds.ColorMLP(16, 3, hidden_dim=16, n_layers=2).cuda()
    base = torch.rand(1, 3, 4, 7, device="cuda")
    extra = torch.rand(1, 3, 4, 9, device="cuda")
    mask = torch.ones(1, 3, 4, dtype=torch.bool, device="cuda")

    if invalid == "shape":
        extra = extra[:, :, :-1]
    elif invalid == "dtype":
        extra = extra.double()
    elif invalid == "device":
        extra = extra.cpu()
    else:
        mask = mask[:, :, :-1]

    with pytest.raises(AssertionError):
        model(base, mask=mask, extra_features=extra)
