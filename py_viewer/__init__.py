"""Python-only DiffSoup viewer.

This package renders web-exported DiffSoup assets without the native
``diffsoupviewer`` extension. Rendering still uses the GPU through PyOpenGL,
but data-loading helpers stay importable without an OpenGL installation.
"""

from .assets import (
    SceneAssets,
    load_exported_scene,
    list_exported_scenes,
    scene_assets_from_arrays,
    scene_assets_from_split_luts,
)

__version__ = "0.1.0"

__all__ = [
    "benchmark",
    "SceneAssets",
    "launch_viewer",
    "load_checkpoint_scene",
    "load_exported_scene",
    "list_exported_scenes",
    "scene_assets_from_arrays",
    "scene_assets_from_split_luts",
]


def __getattr__(name: str):
    if name in {"benchmark", "launch_viewer"}:
        from .api import benchmark, launch_viewer

        return {"benchmark": benchmark, "launch_viewer": launch_viewer}[name]
    if name == "load_checkpoint_scene":
        from .checkpoint import load_checkpoint_scene

        return load_checkpoint_scene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
