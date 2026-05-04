"""GLSL shader resources for the Python DiffSoup viewer."""

from __future__ import annotations

from importlib.resources import files


def _read(name: str) -> str:
    return files(__name__).joinpath(name).read_text(encoding="utf-8")


GEOM_VS = _read("geom.vert.glsl")
GEOM_FS = _read("geom.frag.glsl")
POST_VS = _read("post.vert.glsl")
POST_FS = _read("post.frag.glsl")


__all__ = [
    "GEOM_VS",
    "GEOM_FS",
    "POST_VS",
    "POST_FS",
]
