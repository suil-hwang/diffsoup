"""Contract tests for the SSIM backend used by the training examples."""

from __future__ import annotations

import builtins
from contextlib import nullcontext
import importlib.util
from pathlib import Path

import pytest
import torch

from examples import utils as example_utils


@pytest.mark.parametrize(
    ("requires_grad", "grad_enabled", "expected_train"),
    [
        (True, True, True),
        (False, True, False),
        (True, False, False),
    ],
)
def test_fused_wrapper_uses_valid_padding_and_correct_train_flag(
    monkeypatch: pytest.MonkeyPatch,
    requires_grad: bool,
    grad_enabled: bool,
    expected_train: bool,
) -> None:
    if example_utils.SSIM_BACKEND != "fused_ssim":
        pytest.skip("fused_ssim is not the active backend")

    prediction = torch.rand(1, 3, 15, 17, requires_grad=requires_grad)
    target = torch.rand_like(prediction)
    call: dict[str, object] = {}

    def fake_fused_ssim(
        actual_prediction: torch.Tensor,
        actual_target: torch.Tensor,
        *,
        padding: str,
        train: bool,
    ) -> torch.Tensor:
        call.update(
            prediction=actual_prediction,
            target=actual_target,
            padding=padding,
            train=train,
        )
        return actual_prediction.mean()

    monkeypatch.setattr(example_utils, "_fused_ssim", fake_fused_ssim)
    context = nullcontext() if grad_enabled else torch.no_grad()
    with context:
        result = example_utils.ssim(prediction, target)

    assert result.ndim == 0
    assert call == {
        "prediction": prediction,
        "target": target,
        "padding": "valid",
        "train": expected_train,
    }


def test_reference_fallback_uses_prediction_first_and_unit_data_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the fallback branch even when fused_ssim is installed."""
    original_import = builtins.__import__

    def import_without_fused(name, *args, **kwargs):
        if name == "fused_ssim":
            raise ImportError("blocked for fallback test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fused)
    source = Path(example_utils.__file__)
    spec = importlib.util.spec_from_file_location("utils_ssim_fallback", source)
    assert spec is not None and spec.loader is not None
    fallback = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fallback)
    assert fallback.SSIM_BACKEND == "pytorch_msssim"

    prediction = torch.rand(1, 3, 15, 17)
    target = torch.rand_like(prediction)
    call: dict[str, object] = {}

    def fake_reference(
        actual_prediction: torch.Tensor,
        actual_target: torch.Tensor,
        *,
        data_range: float,
    ) -> torch.Tensor:
        call.update(
            prediction=actual_prediction,
            target=actual_target,
            data_range=data_range,
        )
        return actual_prediction.mean()

    monkeypatch.setattr(fallback, "_pytorch_ssim", fake_reference)
    result = fallback.ssim(prediction, target)
    assert result.ndim == 0
    assert call == {
        "prediction": prediction,
        "target": target,
        "data_range": 1.0,
    }


def _value_and_prediction_grad(
    fn,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample = prediction.detach().clone().requires_grad_(True)
    value = fn(sample, target)
    (gradient,) = torch.autograd.grad(value, sample)
    return value.detach(), gradient.detach()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_value_and_prediction_gradient_match_reference() -> None:
    if example_utils.SSIM_BACKEND != "fused_ssim":
        pytest.skip("fused_ssim is not the active backend")
    reference_ssim = pytest.importorskip("pytorch_msssim").ssim

    torch.manual_seed(20260715)
    prediction = torch.rand(2, 3, 37, 41, device="cuda")
    target = torch.rand_like(prediction)

    actual_value, actual_gradient = _value_and_prediction_grad(
        example_utils.ssim,
        prediction,
        target,
    )
    reference_value, reference_gradient = _value_and_prediction_grad(
        lambda x, y: reference_ssim(x, y, data_range=1.0),
        prediction,
        target,
    )

    torch.testing.assert_close(actual_value, reference_value, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        actual_gradient,
        reference_gradient,
        rtol=2e-4,
        atol=2e-8,
    )
