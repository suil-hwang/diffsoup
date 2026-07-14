"""Pure-Python OpenGL viewer for DiffSoup checkpoints."""

from .scene import SceneData, load_checkpoint_scene
from .viewer import benchmark, launch_scene, launch_viewer

__all__ = [
    "SceneData",
    "benchmark",
    "launch_scene",
    "launch_viewer",
    "load_checkpoint_scene",
]

__version__ = "0.1.0"

