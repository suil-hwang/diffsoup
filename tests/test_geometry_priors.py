from __future__ import annotations

import argparse
from collections.abc import Iterator
from collections import deque
from contextlib import contextmanager, nullcontext
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest
import torch


def _load_source_module(
    name: str,
    path: Path,
    *,
    import_root: Path | None = None,
    isolated_imports: tuple[str, ...] = (),
) -> tuple[ModuleType, dict[str, ModuleType]]:
    """Load one source file without leaking generic import names globally."""
    managed_names = (name, *isolated_imports)
    previous_modules = {
        key: sys.modules[key] for key in managed_names if key in sys.modules
    }
    previous_path = sys.path.copy()
    for key in managed_names:
        sys.modules.pop(key, None)
    if import_root is not None:
        sys.path.insert(0, str(import_root))

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        imported = {}
        for dependency_name in isolated_imports:
            dependency = sys.modules.get(dependency_name)
            assert isinstance(dependency, ModuleType), (
                f"{path}: expected import {dependency_name!r}"
            )
            imported[dependency_name] = dependency
        return module, imported
    finally:
        sys.path[:] = previous_path
        for key in managed_names:
            sys.modules.pop(key, None)
        sys.modules.update(previous_modules)


_REPOSITORY_ROOT = Path(__file__).parents[1]
_EXAMPLES = _REPOSITORY_ROOT / "examples"
priors, _ = _load_source_module(
    "diffsoup_priors_under_test",
    _REPOSITORY_ROOT / "python" / "diffsoup" / "priors.py",
)
prior_cli, _cli_imports = _load_source_module(
    "prepare_geometry_priors_under_test",
    _EXAMPLES / "08_prepare_geometry_priors.py",
    import_root=_EXAMPLES,
    isolated_imports=("utils",),
)
example_utils = _cli_imports["utils"]


@contextmanager
def _example_import_scope() -> Iterator[None]:
    """Expose the exact examples/utils.py only while executing a script."""
    previous_path = sys.path.copy()
    had_utils = "utils" in sys.modules
    previous_utils = sys.modules.get("utils")
    sys.path.insert(0, str(_EXAMPLES))
    sys.modules["utils"] = example_utils
    try:
        yield
    finally:
        sys.path[:] = previous_path
        if had_utils:
            sys.modules["utils"] = previous_utils
        else:
            sys.modules.pop("utils", None)


def _write_unit_prior_layout(
    scene: Path,
    scales: dict[str, float],
    normal_folder: str,
) -> None:
    depth_root = scene / "depth"
    normal_root = scene / normal_folder
    sparse_root = scene / "sparse" / "0"
    depth_root.mkdir(parents=True)
    normal_root.mkdir(parents=True)
    sparse_root.mkdir(parents=True)
    depth = np.full((1, 1), 32768, dtype=np.uint16)
    normal = np.array([[[127, 127, 0]]], dtype=np.uint8)
    for stem in scales:
        iio.imwrite(depth_root / f"{stem}.png", depth)
        iio.imwrite(normal_root / f"{stem}.png", normal)
    (sparse_root / "depth_params.json").write_text(json.dumps({
        stem: {
            "png_scale": scale,
            "offset": 0.0,
            "depth_reliable": True,
            "normal_reliable": True,
            "normal_convention": "camera_xyz_opencv_y_down",
            "validation_median_abs_rel": 0.0,
            "validation_p90_abs_rel": 0.0,
        }
        for stem, scale in scales.items()
    }), encoding="utf-8")


def test_mip360_loader_preserves_per_frame_intrinsics(tmp_path: Path):
    scene = tmp_path / "scene"
    sparse = scene / "sparse" / "0"
    images = scene / "images_4"
    sparse.mkdir(parents=True)
    images.mkdir()
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 8 6 4 5 4 3\n"
        "2 PINHOLE 16 12 12 10 8 6\n",
        encoding="utf-8",
    )
    records = []
    specs = (
        (1, "a.png", 1),
        (2, "b.png", 1),
        (3, "c.png", 2),
        (4, "d.png", 2),
    )
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    for image_id, name, camera_id in specs:
        records.append(
            f"{image_id} 1 0 0 0 0 0 0 {camera_id} {name}"
        )
        records.append("0 0 -1")
        iio.imwrite(images / name, image)
    (sparse / "images.txt").write_text(
        "\n".join(records) + "\n", encoding="utf-8",
    )

    loaded = example_utils.load_mipnerf360_scene(
        str(scene),
        split="train",
        holdout=2,
        downscale=4,
        device=torch.device("cpu"),
    )

    assert loaded["Ks"].shape == (2, 3, 3)
    torch.testing.assert_close(
        loaded["Ks"][0],
        torch.tensor([[4.0, 0.0, 4.0], [0.0, 5.0, 3.0], [0.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(
        loaded["Ks"][1],
        torch.tensor([[6.0, 0.0, 4.0], [0.0, 5.0, 3.0], [0.0, 0.0, 1.0]]),
    )
    assert [frame["camera_id"] for frame in loaded["frames"]] == [1, 2]
def _write_fake_arag(root: Path, source: str) -> Path:
    infer_path = root / "tools" / "infer.py"
    infer_path.parent.mkdir(parents=True)
    infer_path.write_text(source, encoding="utf-8")
    return infer_path


def test_arag_import_uses_exact_path_and_restores_global_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    infer_path = _write_fake_arag(
        tmp_path / "arag",
        """
import sys
from types import ModuleType

XFORMERS_DISABLED = (
    sys.modules.get("xformers") is None
    and sys.modules.get("xformers.ops") is None
)
sys.path.insert(0, "__fake_arag_path__")
for name in ("src", "src._arag_generated", "hubconf", "mono", "mono._arag_generated"):
    sys.modules[name] = ModuleType(name)
""",
    )
    monkeypatch.setattr(prior_cli, "_ARAG_ROOT", infer_path.parents[1])
    names = (
        "xformers",
        "xformers.ops",
        "src",
        "src.existing",
        "hubconf",
        "mono",
        prior_cli._ARAG_INFER_MODULE,
    )
    original_modules = {name: ModuleType(name) for name in names}
    for name, module in original_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    unrelated_infer = ModuleType("tools.infer")
    monkeypatch.setitem(sys.modules, "tools.infer", unrelated_infer)
    for name in ("src._arag_generated", "mono._arag_generated"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    original_path = sys.path.copy()

    with prior_cli._arag_inference_module() as module:
        assert Path(module.__file__).resolve() == infer_path.resolve()
        assert module.__name__ == prior_cli._ARAG_INFER_MODULE
        assert module.XFORMERS_DISABLED
        assert sys.path[0] == "__fake_arag_path__"
        assert sys.modules["src"] is not original_modules["src"]
        assert sys.modules["hubconf"] is not original_modules["hubconf"]
        assert sys.modules["tools.infer"] is unrelated_infer

    assert sys.path == original_path
    for name, module in original_modules.items():
        assert sys.modules[name] is module
    assert "src._arag_generated" not in sys.modules
    assert "mono._arag_generated" not in sys.modules
    assert sys.modules["tools.infer"] is unrelated_infer


def test_arag_import_restores_global_state_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    infer_path = _write_fake_arag(
        tmp_path / "arag",
        """
import sys
from types import ModuleType

sys.path.insert(0, "__failed_arag_path__")
sys.modules["src._arag_failed"] = ModuleType("src._arag_failed")
raise RuntimeError("injected import failure")
""",
    )
    monkeypatch.setattr(prior_cli, "_ARAG_ROOT", infer_path.parents[1])
    original_path = sys.path.copy()
    sentinel = ModuleType("src")
    monkeypatch.setitem(sys.modules, "src", sentinel)
    monkeypatch.delitem(sys.modules, "src._arag_failed", raising=False)

    with pytest.raises(RuntimeError, match="injected import failure"):
        with prior_cli._arag_inference_module():
            pass

    assert sys.path == original_path
    assert sys.modules["src"] is sentinel
    assert "src._arag_failed" not in sys.modules
    assert prior_cli._ARAG_INFER_MODULE not in sys.modules


def test_arag_xyz_normal_returns_canonical_xyz():
    K = torch.tensor(
        [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
    )
    raw_xyz = torch.tensor([[[1.0, 0.0, -1.0]]])
    normal, valid = prior_cli._canonicalize_arag_normals(raw_xyz, K)
    expected = torch.tensor([1.0, 0.0, -1.0])
    expected = expected / expected.norm()
    assert valid.item()
    torch.testing.assert_close(normal[0, 0], expected)


def test_camera_normal_is_face_forwarded_against_pixel_ray():
    K = torch.eye(3)
    raw_xyz = torch.tensor([[[0.0, 0.0, 2.0]]])
    normal, valid = prior_cli._canonicalize_arag_normals(raw_xyz, K)
    assert valid.item()
    torch.testing.assert_close(normal[0, 0], torch.tensor([0.0, 0.0, -1.0]))


def test_refined_normal_keeps_xyz_order_through_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
):
    raw_xyz = torch.tensor([1.0, 2.0, -3.0]).view(1, 3, 1, 1)

    class Model:
        def __call__(self, **_kwargs):
            return None, {
                "depth_pred": torch.ones((1, 1, 2, 2)),
                "normal_pred": raw_xyz.expand(-1, -1, 2, 2),
            }

    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    monkeypatch.setattr(
        prior_cli.torch,
        "autocast",
        lambda **_kwargs: nullcontext(),
    )
    _, refined = prior_cli._run_arag_refiner(
        Model(),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2, 3), dtype=np.float32),
        (2, 2),
    )
    K = torch.tensor(
        [[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
    )
    normal, valid = prior_cli._canonicalize_arag_normals(
        torch.from_numpy(refined), K,
    )

    expected = torch.tensor([1.0, 2.0, -3.0])
    expected = expected / expected.norm()
    assert valid.all()
    torch.testing.assert_close(
        normal, expected.view(1, 1, 3).expand_as(normal),
    )


def test_robust_inverse_depth_affine_rejects_large_outlier():
    relative = np.linspace(0.0, 1.0, 200)
    target = 0.4 * relative + 0.1
    target[0] = 100.0
    slope, shift = prior_cli.fit_inverse_depth_affine(relative, target)
    assert slope == pytest.approx(0.4, rel=2e-3, abs=2e-3)
    assert shift == pytest.approx(0.1, rel=2e-3, abs=2e-3)


@pytest.mark.parametrize("transform", ["identity", "reciprocal"])
def test_depth_fit_selects_the_correct_raw_depth_transform(transform: str):
    point_ids = np.arange(100, dtype=np.int64)
    raw = np.linspace(0.4, 4.0, point_ids.size)
    transformed = raw if transform == "identity" else 1.0 / raw
    target = 0.35 * transformed + 0.08

    fit = prior_cli._fit_depth(raw, target, point_ids)

    assert fit["transform"] == transform
    assert fit["slope"] == pytest.approx(0.35, rel=1e-6, abs=1e-6)
    assert fit["shift"] == pytest.approx(0.08, rel=1e-6, abs=1e-6)


def test_depth_fit_prefers_a_candidate_that_passes_quality_thresholds(
    monkeypatch: pytest.MonkeyPatch,
):
    metrics = iter((
        {"mean_abs_rel": 0.1, "median_abs_rel": 0.1, "p90_abs_rel": 2.0},
        {"mean_abs_rel": 0.2, "median_abs_rel": 0.2, "p90_abs_rel": 0.5},
    ))
    monkeypatch.setattr(
        prior_cli, "fit_inverse_depth_affine", lambda *_args: (1.0, 0.0),
    )
    monkeypatch.setattr(
        prior_cli, "_depth_metrics", lambda *_args: next(metrics),
    )

    fit = prior_cli._fit_depth(
        np.linspace(0.5, 4.0, 100),
        np.linspace(0.1, 1.0, 100),
        np.arange(100, dtype=np.int64),
    )

    assert fit["transform"] == "reciprocal"


def test_depth_fit_does_not_hide_affine_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    point_ids = np.arange(100, dtype=np.int64)
    raw = np.linspace(0.4, 4.0, point_ids.size)
    target = 0.35 * raw + 0.08
    failure = ValueError("unexpected affine failure")

    def fail(*_args):
        raise failure

    monkeypatch.setattr(prior_cli, "fit_inverse_depth_affine", fail)
    with pytest.raises(ValueError) as caught:
        prior_cli._fit_depth(raw, target, point_ids)
    assert caught.value is failure


def test_coarse_pair_normalizes_layout_dtype_and_contiguity():
    depth = np.arange(4, dtype=np.float64).reshape(2, 2, 1)
    normal_chw = np.arange(12, dtype=np.float64).reshape(3, 2, 2)

    normalized_depth, normalized_normal = prior_cli._normalize_coarse_pair(
        depth, normal_chw, (2, 2),
    )

    assert normalized_depth.shape == (2, 2)
    assert normalized_normal.shape == (2, 2, 3)
    assert normalized_depth.dtype == np.float32
    assert normalized_normal.dtype == np.float32
    assert normalized_depth.flags.c_contiguous
    assert normalized_normal.flags.c_contiguous
    np.testing.assert_array_equal(normalized_depth, depth[..., 0])
    np.testing.assert_array_equal(
        normalized_normal, np.moveaxis(normal_chw, 0, -1),
    )

    with pytest.raises(RuntimeError, match="coarse output shapes"):
        prior_cli._normalize_coarse_pair(
            np.ones((3, 2)), normal_chw, (2, 2),
        )


def test_coarse_inference_returns_normalized_memory_spool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    image_path = tmp_path / "frame.png"
    iio.imwrite(image_path, np.zeros((2, 3, 3), dtype=np.uint8))
    scene = SimpleNamespace(
        records=(object(),),
        image_paths=(image_path,),
        image_size=(2, 3),
    )
    events = []

    class InferModule:
        def load_dav2_model(self, *_args):
            events.append("load_depth")
            return object()

        def load_metric3d_model(self, *_args):
            events.append("load_normal")
            return object()

        def run_dav2(self, *_args, **_kwargs):
            return np.arange(6, dtype=np.float64).reshape(2, 3, 1)

        def run_metric3d(self, *_args, **_kwargs):
            return np.arange(18, dtype=np.float64).reshape(3, 2, 3)

    monkeypatch.setattr(prior_cli.torch.cuda, "is_available", lambda: False)
    coarse = prior_cli._infer_coarse(InferModule(), scene)

    assert isinstance(coarse, deque)
    assert len(coarse) == 1
    depth, normal = coarse.popleft()
    assert depth.shape == (2, 3) and depth.dtype == np.float32
    assert normal.shape == (2, 3, 3) and normal.dtype == np.float32
    assert depth.flags.c_contiguous and normal.flags.c_contiguous
    assert events == ["load_depth", "load_normal"]


def test_refined_stage_writes_canonical_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    image_path = tmp_path / "frame.png"
    iio.imwrite(image_path, np.zeros((2, 3, 3), dtype=np.uint8))
    record = SimpleNamespace(name="frame.png", camera_id=1)
    scene = SimpleNamespace(
        image_size=(2, 3),
        records=(record,),
        image_paths=(image_path,),
        cameras={1: {}},
        points={},
    )
    events = []

    class InferModule:
        def build_patch_processor(self, height, width, patch_split):
            events.append(("processor", height, width, patch_split))
            return object(), height, width

        def build_urgt_model(self, *_args, **_kwargs):
            events.append(("model",))
            return object()

    monkeypatch.setattr(
        prior_cli,
        "_run_arag_refiner",
        lambda *_args: (
            np.ones((2, 3), dtype=np.float32),
            np.broadcast_to(
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                (2, 3, 3),
            ).copy(),
        ),
    )
    monkeypatch.setattr(
        prior_cli,
        "_sparse_samples",
        lambda *_args: (
            np.empty(0), np.empty(0), np.empty(0, dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        prior_cli,
        "_fit_depth",
        lambda *_args: {
            "transform": "identity",
            "slope": 1.0,
            "shift": 0.0,
            "num_fit": 32,
            "num_validation": 8,
            "validation_metrics": {
                "mean_abs_rel": 0.0,
                "median_abs_rel": 0.0,
                "p90_abs_rel": 0.0,
            },
        },
    )
    monkeypatch.setattr(
        prior_cli,
        "_scaled_intrinsics",
        lambda *_args: np.array(
            [[10.0, 0.0, 1.5], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    coarse = deque([(
        np.ones((2, 3), dtype=np.float32),
        np.ones((2, 3, 3), dtype=np.float32),
    )])
    staging = tmp_path / "staging"

    report = prior_cli._stage_refined_priors(
        InferModule(), scene, coarse, staging, "normals_4",
        tmp_path / "checkpoint.pth", (1, 1),
    )

    assert report == {
        "frames": 1,
        "depth_reliable_frames": 1,
        "normal_reliable_frames": 1,
        "unreliable_frames": 0,
    }
    assert not coarse
    assert events == [("processor", 2, 3, (1, 1)), ("model",)]
    depth = iio.imread(staging / "depth" / "frame.png")
    normal = iio.imread(staging / "normals_4" / "frame.png")
    assert depth.dtype == np.uint16 and np.all(depth == 65535)
    assert np.all(normal == np.array([127, 127, 0], dtype=np.uint8))
    params = json.loads(
        (staging / "sparse" / "0" / "depth_params.json").read_text()
    )
    frame_params = params["frame"]
    assert frame_params["offset"] == 0.0
    assert frame_params["png_scale"] == pytest.approx(65536.0 / 65535.0)
    assert "scale" not in frame_params
    assert frame_params["depth_reliable"] is True
    assert frame_params["normal_reliable"] is True
    assert frame_params["fit_transform"] == "identity"
    assert frame_params["fit_slope"] == 1.0
    assert frame_params["fit_shift"] == 0.0
    assert frame_params["validation_median_abs_rel"] == 0.0
    assert frame_params["validation_p90_abs_rel"] == 0.0
    assert frame_params["normal_convention"] == "camera_xyz_opencv_y_down"

    store = priors.GeometryPriorStore(
        staging, ["frame.png"], (2, 3), downscale=4,
    )
    samples = store.sample_joint_uniform(
        [0], 6, np.random.default_rng(0), "cpu",
    )
    assert samples.depth_valid.all()
    assert samples.normal_valid.all()
    torch.testing.assert_close(
        samples.inverse_camera_z, torch.ones_like(samples.inverse_camera_z),
    )


def test_refined_stage_marks_bad_modalities_per_view_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    image_paths = []
    records = []
    for name in ("bad_depth.png", "bad_normal.png"):
        image_path = tmp_path / name
        iio.imwrite(image_path, np.zeros((2, 3, 3), dtype=np.uint8))
        image_paths.append(image_path)
        records.append(SimpleNamespace(name=name, camera_id=1))
    scene = SimpleNamespace(
        image_size=(2, 3),
        records=tuple(records),
        image_paths=tuple(image_paths),
        cameras={1: {}},
        points={},
    )

    class InferModule:
        @staticmethod
        def build_patch_processor(height, width, _patch_split):
            return object(), height, width

        @staticmethod
        def build_urgt_model(*_args, **_kwargs):
            return object()

    monkeypatch.setattr(
        prior_cli,
        "_run_arag_refiner",
        lambda *_args: (
            np.ones((2, 3), dtype=np.float32),
            np.broadcast_to(
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
                (2, 3, 3),
            ).copy(),
        ),
    )
    monkeypatch.setattr(
        prior_cli,
        "_sparse_samples",
        lambda *_args: (
            np.empty(0), np.empty(0), np.empty(0, dtype=np.int64),
        ),
    )
    fit_calls = 0

    def fit_per_view(*_args):
        nonlocal fit_calls
        fit_calls += 1
        if fit_calls == 1:
            raise ValueError("not enough sparse points")
        return {
            "transform": "identity",
            "slope": 1.0,
            "shift": 0.0,
            "validation_metrics": {
                "mean_abs_rel": 0.0,
                "median_abs_rel": 0.0,
                "p90_abs_rel": 0.0,
            },
        }

    monkeypatch.setattr(prior_cli, "_fit_depth", fit_per_view)
    monkeypatch.setattr(
        prior_cli,
        "_scaled_intrinsics",
        lambda *_args: np.eye(3, dtype=np.float64),
    )
    normal_calls = 0

    def normal_per_view(*_args):
        nonlocal normal_calls
        normal_calls += 1
        normal = torch.zeros((2, 3, 3), dtype=torch.float32)
        normal[..., 2] = -1.0
        valid = torch.ones((2, 3), dtype=torch.bool)
        if normal_calls == 2:
            normal.zero_()
            valid.zero_()
        return normal, valid

    monkeypatch.setattr(
        prior_cli, "_canonicalize_arag_normals", normal_per_view,
    )
    coarse = deque([
        (
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3, 3), dtype=np.float32),
        )
        for _ in records
    ])
    staging = tmp_path / "staging"

    report = prior_cli._stage_refined_priors(
        InferModule(), scene, coarse, staging, "normals_4",
        tmp_path / "checkpoint.pth", (1, 1),
    )

    assert report == {
        "frames": 2,
        "depth_reliable_frames": 1,
        "normal_reliable_frames": 1,
        "unreliable_frames": 2,
    }
    assert not coarse
    assert not iio.imread(staging / "depth" / "bad_depth.png").any()
    assert np.all(
        iio.imread(staging / "normals_4" / "bad_depth.png")
        == np.array([127, 127, 0], dtype=np.uint8)
    )
    assert iio.imread(staging / "depth" / "bad_normal.png").any()
    assert np.all(
        iio.imread(staging / "normals_4" / "bad_normal.png") == 127
    )
    params = json.loads(
        (staging / "sparse" / "0" / "depth_params.json").read_text()
    )
    assert params["bad_depth"]["depth_reliable"] is False
    assert params["bad_depth"]["normal_reliable"] is True
    assert params["bad_normal"]["depth_reliable"] is True
    assert params["bad_normal"]["normal_reliable"] is False
    assert "failure_reasons" in params["bad_depth"]
    assert "failure_reasons" in params["bad_normal"]


def test_tiny_positive_depth_keeps_a_positive_png_scale():
    inverse_depth = np.full((2, 2), 1e-15, dtype=np.float32)
    encoded, scale = prior_cli._encode_depth_png(
        inverse_depth, np.ones((2, 2), dtype=np.bool_),
    )

    assert scale > 0.0
    assert np.all(encoded == np.iinfo(np.uint16).max)


def test_scene_prior_store_decodes_canonical_layout(tmp_path: Path):
    scene = tmp_path / "scene"
    depth_root = scene / "depth"
    normal_root = scene / "normals_4"
    sparse_root = scene / "sparse" / "0"
    depth_root.mkdir(parents=True)
    normal_root.mkdir(parents=True)
    sparse_root.mkdir(parents=True)

    depth_a = np.array([[32768, 0], [16384, 8192]], dtype=np.uint16)
    depth_b = np.array([[16384, 8192], [4096, 2048]], dtype=np.uint16)
    normal = np.empty((2, 2, 3), dtype=np.uint8)
    normal[...] = np.array([200, 80, 30], dtype=np.uint8)
    normal[0, 1] = 127
    iio.imwrite(depth_root / "a.png", depth_a)
    iio.imwrite(depth_root / "b.png", depth_b)
    iio.imwrite(normal_root / "a.png", normal)
    iio.imwrite(normal_root / "b.png", normal)
    (sparse_root / "depth_params.json").write_text(json.dumps({
        "a": {
            "png_scale": 2.0,
            "offset": 0.25,
            "depth_reliable": True,
            "normal_reliable": True,
            "normal_convention": "camera_xyz_opencv_y_down",
            "validation_median_abs_rel": 0.05,
            "validation_p90_abs_rel": 0.15,
        },
        "b": {
            "png_scale": 4.0,
            "offset": 0.5,
            "depth_reliable": True,
            "normal_reliable": True,
            "normal_convention": "camera_xyz_opencv_y_down",
            "validation_median_abs_rel": 0.05,
            "validation_p90_abs_rel": 0.15 * np.sqrt(2.0),
        },
    }))

    store = priors.GeometryPriorStore(
        scene,
        ["b.JPG", "a.JPG"],
        (2, 2),
        downscale=4,
    )
    assert store.view_names == ("b.JPG", "a.JPG")
    assert store.image_size == (2, 2)

    samples = store.sample_joint_uniform(
        [0, 1], 8, np.random.default_rng(4), "cpu",
    )
    assert samples.pixels_b_y_x.shape == (16, 3)
    assert (samples.pixels_b_y_x[:8, 0] == 0).all()
    assert (samples.pixels_b_y_x[8:, 0] == 1).all()
    reference_rng = np.random.default_rng(4)
    reference_flat = np.stack([
        reference_rng.integers(0, 4, size=8, dtype=np.int64)
        for _ in range(2)
    ])
    torch.testing.assert_close(
        samples.pixels_b_y_x[:, 1],
        torch.from_numpy((reference_flat // 2).reshape(-1)),
    )
    torch.testing.assert_close(
        samples.pixels_b_y_x[:, 2],
        torch.from_numpy((reference_flat % 2).reshape(-1)),
    )
    assert torch.isfinite(samples.inverse_camera_z).all()
    assert torch.isfinite(samples.normal_camera).all()
    for local_batch, encoded, scale, offset in (
        (0, depth_b, 4.0, 0.5),
        (1, depth_a, 2.0, 0.25),
    ):
        selected = samples.pixels_b_y_x[:, 0] == local_batch
        y = samples.pixels_b_y_x[selected, 1].numpy()
        x = samples.pixels_b_y_x[selected, 2].numpy()
        selected_depth = encoded[y, x]
        expected = selected_depth.astype(np.float32) / 65536.0 * scale + offset
        expected = np.where(selected_depth > 0, expected, 0.0).astype(np.float32)
        torch.testing.assert_close(
            samples.inverse_camera_z[selected], torch.from_numpy(expected),
        )
        torch.testing.assert_close(
            samples.depth_valid[selected], torch.from_numpy(selected_depth > 0),
        )
        selected_normal = normal[y, x]
        normal_valid = ~np.all(selected_normal == 127, axis=-1)
        expected_normal = selected_normal.astype(np.float32) / 255.0 * 2.0 - 1.0
        expected_normal /= np.maximum(
            np.linalg.norm(expected_normal, axis=-1, keepdims=True), 1e-8,
        )
        expected_normal = np.where(
            normal_valid[:, None], expected_normal, 0.0,
        ).astype(np.float32)
        torch.testing.assert_close(
            samples.normal_camera[selected], torch.from_numpy(expected_normal),
        )
        torch.testing.assert_close(
            samples.normal_valid[selected], torch.from_numpy(normal_valid),
        )
    empty = store.sample_joint_uniform(
        [], 8, np.random.default_rng(5), "cpu", dtype=torch.float64,
    )
    assert empty.pixels_b_y_x.shape == (0, 3)
    assert empty.inverse_camera_z.shape == (0,)
    assert empty.inverse_camera_z.dtype == torch.float64
    assert empty.normal_camera.shape == (0, 3)
    with pytest.raises(AssertionError):
        store.sample_joint_uniform(
            [0], 4, np.random.default_rng(1), "cpu", dtype=torch.int64,
        )
    with pytest.raises(AssertionError, match="integer sequence"):
        store.sample_joint_uniform(
            [0.5], 4, np.random.default_rng(1), "cpu",
        )
    with pytest.raises(AssertionError, match="image_size"):
        priors.GeometryPriorStore(
            scene, ["a.JPG"], (1.5, 1), downscale=4,
        )


def test_scene_prior_store_accepts_legacy_scale_alias(tmp_path: Path):
    scene = tmp_path / "scene"
    _write_unit_prior_layout(scene, {"frame": 2.0}, "normals_4")
    params_path = scene / "sparse" / "0" / "depth_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["frame"]["scale"] = params["frame"].pop("png_scale")
    params_path.write_text(json.dumps(params), encoding="utf-8")

    store = priors.GeometryPriorStore(
        scene, ["frame.jpg"], (1, 1), downscale=4, load_normal=False,
    )
    samples = store.sample_joint_uniform(
        [0], 1, np.random.default_rng(0), "cpu",
    )

    assert samples.depth_valid.item()
    assert samples.inverse_camera_z.item() == pytest.approx(1.0)


def test_png_scale_takes_precedence_over_legacy_scale_alias(tmp_path: Path):
    scene = tmp_path / "scene"
    _write_unit_prior_layout(scene, {"frame": 2.0}, "normals_4")
    params_path = scene / "sparse" / "0" / "depth_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["frame"]["scale"] = 200.0
    params_path.write_text(json.dumps(params), encoding="utf-8")

    store = priors.GeometryPriorStore(
        scene, ["frame.jpg"], (1, 1), downscale=4, load_normal=False,
    )
    samples = store.sample_joint_uniform(
        [0], 1, np.random.default_rng(0), "cpu",
    )

    assert samples.depth_valid.item()
    assert samples.inverse_camera_z.item() == pytest.approx(1.0)


@pytest.mark.parametrize(("field", "value"), (
    ("png_scale", 0.0),
    ("png_scale", "inf"),
    ("offset", "nan"),
))
def test_prior_store_rejects_invalid_depth_decode_parameters(
    tmp_path: Path,
    field: str,
    value: object,
):
    scene = tmp_path / "scene"
    _write_unit_prior_layout(scene, {"frame": 2.0}, "normals_4")
    params_path = scene / "sparse" / "0" / "depth_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["frame"][field] = value
    params_path.write_text(json.dumps(params), encoding="utf-8")

    with pytest.raises(AssertionError):
        priors.GeometryPriorStore(
            scene, ["frame.jpg"], (1, 1), downscale=4, load_normal=False,
        )


@pytest.mark.parametrize("field", ("depth_reliable", "normal_reliable"))
def test_prior_store_requires_boolean_reliability_flags(
    tmp_path: Path,
    field: str,
):
    scene = tmp_path / "scene"
    _write_unit_prior_layout(scene, {"frame": 2.0}, "normals_4")
    params_path = scene / "sparse" / "0" / "depth_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["frame"][field] = 1
    params_path.write_text(json.dumps(params), encoding="utf-8")

    with pytest.raises(AssertionError):
        priors.GeometryPriorStore(
            scene, ["frame.jpg"], (1, 1), downscale=4,
        )


def test_prior_store_rejects_unknown_normal_convention(tmp_path: Path):
    scene = tmp_path / "scene"
    _write_unit_prior_layout(scene, {"frame": 2.0}, "normals_4")
    params_path = scene / "sparse" / "0" / "depth_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["frame"]["normal_convention"] = "world_xyz"
    params_path.write_text(json.dumps(params), encoding="utf-8")

    with pytest.raises(AssertionError, match="unsupported normal_convention"):
        priors.GeometryPriorStore(
            scene, ["frame.jpg"], (1, 1), downscale=4, load_depth=False,
        )


def test_scene_publish_restores_previous_layout_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = tmp_path / "scene"
    staging = tmp_path / "staging"
    _write_unit_prior_layout(scene, {"old": 2.0}, "normals_4")
    _write_unit_prior_layout(staging, {"new": 4.0}, "normals_4")
    old_paths = (
        scene / "depth" / "old.png",
        scene / "normals_4" / "old.png",
        scene / "sparse" / "0" / "depth_params.json",
    )
    old_contents = [path.read_bytes() for path in old_paths]

    real_replace = prior_cli.os.replace
    failed = False

    def fail_new_normal_once(source, destination):
        nonlocal failed
        if not failed and Path(source).resolve() == (staging / "normals_4").resolve():
            failed = True
            raise OSError("injected publish failure")
        real_replace(source, destination)

    monkeypatch.setattr(prior_cli.os, "replace", fail_new_normal_once)
    with pytest.raises(OSError, match="injected publish failure"):
        prior_cli._publish_scene_priors(
            scene, staging, "normals_4", overwrite=True,
        )

    assert [path.read_bytes() for path in old_paths] == old_contents
    assert not (scene / "depth" / "new.png").exists()
    assert (staging / "depth" / "new.png").is_file()
    assert (staging / "normals_4" / "new.png").is_file()
    assert (staging / "sparse" / "0" / "depth_params.json").is_file()
    assert not (staging / ".publish-backup").exists()


def test_scene_publish_keeps_backup_when_rollback_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = tmp_path / "scene"
    staging = tmp_path / "staging"
    _write_unit_prior_layout(scene, {"old": 2.0}, "normals_4")
    _write_unit_prior_layout(staging, {"new": 4.0}, "normals_4")
    real_replace = prior_cli.os.replace

    def fail_publish_and_rollback(source, destination):
        source = Path(source).resolve()
        destination = Path(destination).resolve()
        if source == (staging / "normals_4").resolve():
            raise OSError("injected publish failure")
        if (
            source == (scene / "depth").resolve()
            and destination == (staging / "depth").resolve()
        ):
            raise OSError("injected rollback failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        prior_cli.os, "replace", fail_publish_and_rollback,
    )
    with pytest.raises(RuntimeError, match="publish and rollback both failed"):
        prior_cli._publish_scene_priors(
            scene, staging, "normals_4", overwrite=True,
        )

    backup = staging / ".publish-backup"
    assert not (scene / "sparse" / "0" / "depth_params.json").exists()
    assert (backup / "sparse" / "0" / "depth_params.json").is_file()


def test_scene_publish_installs_fresh_layout_and_supports_overwrite(tmp_path: Path):
    scene = tmp_path / "scene"
    first_staging = tmp_path / "first_staging"
    _write_unit_prior_layout(first_staging, {"first": 2.0}, "normals_4")

    prior_cli._publish_scene_priors(
        scene, first_staging, "normals_4", overwrite=False,
    )
    assert (scene / "depth" / "first.png").is_file()
    assert (scene / "normals_4" / "first.png").is_file()
    assert json.loads(
        (scene / "sparse" / "0" / "depth_params.json").read_text()
    ) == {"first": {
        "offset": 0.0,
        "png_scale": 2.0,
        "depth_reliable": True,
        "normal_reliable": True,
        "normal_convention": "camera_xyz_opencv_y_down",
        "validation_median_abs_rel": 0.0,
        "validation_p90_abs_rel": 0.0,
    }}

    blocked_staging = tmp_path / "blocked_staging"
    _write_unit_prior_layout(blocked_staging, {"blocked": 3.0}, "normals_4")
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        prior_cli._publish_scene_priors(
            scene, blocked_staging, "normals_4", overwrite=False,
        )
    assert (scene / "depth" / "first.png").is_file()
    assert (blocked_staging / "depth" / "blocked.png").is_file()

    second_staging = tmp_path / "second_staging"
    _write_unit_prior_layout(second_staging, {"second": 4.0}, "normals_4")
    prior_cli._publish_scene_priors(
        scene, second_staging, "normals_4", overwrite=True,
    )
    assert not (scene / "depth" / "first.png").exists()
    assert not (scene / "normals_4" / "first.png").exists()
    assert (scene / "depth" / "second.png").is_file()
    assert (scene / "normals_4" / "second.png").is_file()
    assert json.loads(
        (scene / "sparse" / "0" / "depth_params.json").read_text()
    ) == {"second": {
        "offset": 0.0,
        "png_scale": 4.0,
        "depth_reliable": True,
        "normal_reliable": True,
        "normal_convention": "camera_xyz_opencv_y_down",
        "validation_median_abs_rel": 0.0,
        "validation_p90_abs_rel": 0.0,
    }}
    assert not (second_staging / ".publish-backup").exists()


def test_prepare_preserves_staging_when_publish_rollback_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = tmp_path / "scene"
    checkpoint = tmp_path / "model.pth"
    scene.mkdir()
    checkpoint.touch()
    scene_data = SimpleNamespace(image_folder="images_4")

    monkeypatch.setattr(prior_cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prior_cli, "_load_scene", lambda *_args: scene_data)
    monkeypatch.setattr(prior_cli, "_ensure_publishable", lambda *_args: None)
    monkeypatch.setattr(
        prior_cli,
        "_arag_inference_module",
        lambda: nullcontext(ModuleType("fake_arag")),
    )
    monkeypatch.setattr(prior_cli, "_infer_coarse", lambda *_args: deque())
    monkeypatch.setattr(
        prior_cli,
        "_stage_refined_priors",
        lambda *_args: {"frames": 1},
    )

    def fail_publish(_scene, staging, *_args):
        (Path(staging) / ".publish-backup").mkdir()
        raise RuntimeError("rollback failed")

    monkeypatch.setattr(prior_cli, "_publish_scene_priors", fail_publish)
    args = SimpleNamespace(
        scene_root=scene,
        checkpoint=checkpoint,
        downscale=4,
        patch_split=(2, 2),
        overwrite=True,
    )

    with pytest.raises(RuntimeError, match="rollback failed"):
        prior_cli.prepare_arag_scene(args)

    staging = list(scene.glob(".diffsoup-priors-*"))
    assert len(staging) == 1
    assert (staging[0] / ".publish-backup").is_dir()


def test_prepare_removes_staging_after_regular_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = tmp_path / "scene"
    checkpoint = tmp_path / "model.pth"
    scene.mkdir()
    checkpoint.touch()
    scene_data = SimpleNamespace(image_folder="images_4")

    monkeypatch.setattr(prior_cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prior_cli, "_load_scene", lambda *_args: scene_data)
    monkeypatch.setattr(prior_cli, "_ensure_publishable", lambda *_args: None)
    monkeypatch.setattr(
        prior_cli,
        "_arag_inference_module",
        lambda: nullcontext(ModuleType("fake_arag")),
    )
    monkeypatch.setattr(prior_cli, "_infer_coarse", lambda *_args: deque())

    def fail_stage(*_args):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(prior_cli, "_stage_refined_priors", fail_stage)
    args = SimpleNamespace(
        scene_root=scene,
        checkpoint=checkpoint,
        downscale=4,
        patch_split=(2, 2),
        overwrite=False,
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        prior_cli.prepare_arag_scene(args)

    assert not list(scene.glob(".diffsoup-priors-*"))


def test_prepare_shares_one_scoped_arag_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scene = tmp_path / "scene"
    checkpoint = tmp_path / "model.pth"
    scene.mkdir()
    checkpoint.touch()
    scene_data = SimpleNamespace(image_folder="images_4")
    infer_module = ModuleType("fake_arag")
    coarse_priors = deque([(
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1, 3), dtype=np.float32),
    )])
    events = []

    @contextmanager
    def arag_context():
        events.append("enter")
        try:
            yield infer_module
        finally:
            events.append("exit")

    def infer_coarse(module, *_args):
        assert module is infer_module
        events.append("coarse")
        return coarse_priors

    def stage_refined(module, _scene, received, *_args):
        assert module is infer_module
        assert received is coarse_priors
        received.popleft()
        events.append("refined")
        return {"frames": 1}

    def publish(*_args):
        events.append("publish")

    monkeypatch.setattr(prior_cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(prior_cli, "_load_scene", lambda *_args: scene_data)
    monkeypatch.setattr(prior_cli, "_ensure_publishable", lambda *_args: None)
    monkeypatch.setattr(prior_cli, "_arag_inference_module", arag_context)
    monkeypatch.setattr(prior_cli, "_infer_coarse", infer_coarse)
    monkeypatch.setattr(prior_cli, "_stage_refined_priors", stage_refined)
    monkeypatch.setattr(prior_cli, "_publish_scene_priors", publish)
    args = SimpleNamespace(
        scene_root=scene,
        checkpoint=checkpoint,
        downscale=4,
        patch_split=(2, 2),
        overwrite=False,
    )

    assert prior_cli.prepare_arag_scene(args) == {"frames": 1}
    assert events == ["enter", "coarse", "refined", "exit", "publish"]
    assert not coarse_priors


def test_normal_canonicalization_rejects_invalid_calibration():
    raw = torch.tensor([[[0.0, 0.0, -1.0]]])
    K = torch.eye(3)
    K[0, 0] = 0.0
    with pytest.raises(ValueError, match="focal"):
        prior_cli._canonicalize_arag_normals(raw, K)


def test_affine_fit_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="equal size"):
        prior_cli.fit_inverse_depth_affine(
            np.array([0.1, 0.2]), np.array([0.3]),
        )


def test_png_scale_magnitude_does_not_disable_depth_samples(tmp_path: Path):
    scene = tmp_path / "scene"
    _write_unit_prior_layout(
        scene,
        {"a": 1.0, "b": 1.0, "outlier": 10.0},
        "normals_4",
    )
    store = priors.GeometryPriorStore(
        scene,
        ["a.jpg", "b.jpg", "outlier.jpg"],
        (1, 1),
        downscale=4,
    )

    samples = store.sample_joint_uniform(
        [0, 2], 1, np.random.default_rng(0), "cpu",
    )
    assert samples.depth_valid.tolist() == [True, True]
    assert samples.inverse_camera_z.tolist() == pytest.approx([0.5, 5.0])
    assert samples.normal_valid.tolist() == [True, True]


def test_explicit_reliability_flags_mask_depth_and_normal_independently(
    tmp_path: Path,
):
    scene = tmp_path / "scene"
    _write_unit_prior_layout(
        scene,
        {"good": 2.0, "bad_depth": 2.0, "bad_normal": 2.0},
        "normals_4",
    )
    params_path = scene / "sparse" / "0" / "depth_params.json"
    params = json.loads(params_path.read_text())
    params["bad_depth"]["depth_reliable"] = False
    params["bad_normal"]["normal_reliable"] = False
    params_path.write_text(json.dumps(params))

    store = priors.GeometryPriorStore(
        scene,
        ["good.jpg", "bad_depth.jpg", "bad_normal.jpg"],
        (1, 1),
        downscale=4,
    )
    samples = store.sample_joint_uniform(
        [0, 1, 2], 1, np.random.default_rng(0), "cpu",
    )

    assert samples.depth_valid.tolist() == [True, False, True]
    assert samples.normal_valid.tolist() == [True, True, False]


def test_prior_store_loads_only_the_enabled_modalities(tmp_path: Path):
    normal_scene = tmp_path / "normal_scene"
    normal_root = normal_scene / "normals_4"
    normal_root.mkdir(parents=True)
    iio.imwrite(
        normal_root / "frame.png",
        np.array([[[127, 127, 0]]], dtype=np.uint8),
    )
    normal_store = priors.GeometryPriorStore(
        normal_scene,
        ["frame.jpg"],
        (1, 1),
        downscale=4,
        load_depth=False,
        load_normal=True,
    )
    normal_samples = normal_store.sample_joint_uniform(
        [0], 1, np.random.default_rng(0), "cpu",
    )
    assert not normal_samples.depth_valid.item()
    assert normal_samples.normal_valid.item()

    depth_scene = tmp_path / "depth_scene"
    depth_root = depth_scene / "depth"
    sparse_root = depth_scene / "sparse" / "0"
    depth_root.mkdir(parents=True)
    sparse_root.mkdir(parents=True)
    iio.imwrite(
        depth_root / "frame.png",
        np.array([[32768]], dtype=np.uint16),
    )
    (sparse_root / "depth_params.json").write_text(json.dumps({
        "frame": {
            "png_scale": 2.0,
            "offset": 0.0,
            "depth_reliable": True,
        },
    }))
    depth_store = priors.GeometryPriorStore(
        depth_scene,
        ["frame.jpg"],
        (1, 1),
        downscale=4,
        load_depth=True,
        load_normal=False,
    )
    depth_samples = depth_store.sample_joint_uniform(
        [0], 1, np.random.default_rng(0), "cpu",
    )
    assert depth_samples.depth_valid.item()
    assert not depth_samples.normal_valid.item()


def test_sparse_samples_respect_colmap_half_pixel_coordinates():
    record = prior_cli.ColmapImage(
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        camera_id=1,
        name="frame.png",
        points_xy=np.array([[100.5, 80.5]]),
        point3d_ids=np.array([7], dtype=np.int64),
    )
    camera = {"w": 400, "h": 320}
    yy, xx = np.mgrid[:80, :100]
    raw_depth = (100.0 * yy + xx).astype(np.float64)

    relative, target, point_ids = prior_cli._sparse_samples(
        record,
        camera,
        {7: np.array([0.0, 0.0, 2.0])},
        raw_depth,
    )

    expected_x = 100.5 * 0.25 - 0.5
    expected_y = 80.5 * 0.25 - 0.5
    assert relative.tolist() == pytest.approx([100.0 * expected_y + expected_x])
    assert target.tolist() == pytest.approx([0.5])
    assert point_ids.tolist() == [7]


@pytest.mark.parametrize("downscale", [0, 1])
def test_full_resolution_priors_use_normals_folder(
    tmp_path: Path,
    downscale: int,
):
    scene = tmp_path / f"scene_{downscale}"
    _write_unit_prior_layout(scene, {"frame": 2.0}, "normals")
    store = priors.GeometryPriorStore(
        scene,
        ["frame.jpg"],
        (1, 1),
        downscale=downscale,
    )

    samples = store.sample_joint_uniform(
        [0], 1, np.random.default_rng(0), "cpu",
    )
    assert samples.depth_valid.item()
    assert samples.normal_valid.item()


def test_geometry_prior_cli_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}
    parse_args = argparse.ArgumentParser.parse_args

    class ParsingStopped(Exception):
        pass

    def capture_parser(parser, *_args, **_kwargs):
        captured["parser"] = parser
        raise ParsingStopped

    unrelated_utils = ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", unrelated_utils)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_parser)
    with _example_import_scope():
        with pytest.raises(ParsingStopped):
            runpy.run_path(
                str(_EXAMPLES / "08_prepare_geometry_priors.py"),
                run_name="__main__",
            )
    assert sys.modules["utils"] is unrelated_utils

    args = parse_args(captured["parser"], ["--scene-root", "scene"])
    assert not hasattr(args, "command")
    assert args.downscale == 4
    assert tuple(args.patch_split) == (2, 2)
    assert not args.overwrite
    repository_root = Path(prior_cli.__file__).resolve().parents[1]
    assert Path(args.checkpoint) == (
        repository_root
        / "submodules"
        / "arag"
        / "work_dir"
        / "ckpts"
        / "ckpt_promask_best.pth"
    )


def test_log_linear_schedule_has_exact_endpoints_and_monotonic_decay():
    steps = (1, 2_500, 5_000, 7_500, 10_000)
    values = [
        example_utils.log_linear_schedule(0.01, 0.001, step, 10_000)
        for step in steps
    ]

    assert values[0] == pytest.approx(0.01)
    assert values[-1] == pytest.approx(0.001)
    assert all(left > right for left, right in zip(values, values[1:]))
    assert example_utils.log_linear_schedule(
        0.01, 0.001, 0, 10_000,
    ) == pytest.approx(0.01)
    assert example_utils.log_linear_schedule(
        0.01, 0.001, 20_000, 10_000,
    ) == pytest.approx(0.001)


def test_training_cli_rejects_configurable_prior_lambdas(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}
    parse_args = argparse.ArgumentParser.parse_args

    class ParsingStopped(Exception):
        pass

    def capture_parser(parser, *_args, **_kwargs):
        captured["parser"] = parser
        raise ParsingStopped

    unrelated_utils = ModuleType("utils")
    monkeypatch.setitem(sys.modules, "utils", unrelated_utils)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture_parser)
    with _example_import_scope():
        with pytest.raises(ParsingStopped):
            runpy.run_path(
                str(_EXAMPLES / "02_mip360_test.py"),
                run_name="__main__",
            )
    assert sys.modules["utils"] is unrelated_utils

    parser = captured["parser"]
    args = parse_args(parser, ["--scene_root", "scene"])
    assert not hasattr(args, "lambda_normal_prior")
    assert not hasattr(args, "lambda_depth_prior")
    assert not hasattr(args, "lambda_depth_prior_final")
    for option in (
        "--lambda_normal_prior",
        "--lambda_depth_prior",
        "--lambda_depth_prior_final",
        "--normal_prior_weight",
        "--depth_prior_weight",
    ):
        with pytest.raises(SystemExit):
            parse_args(parser, [option, "0.01"])
