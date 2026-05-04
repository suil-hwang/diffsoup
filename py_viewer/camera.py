"""Orbit camera math for the Python DiffSoup viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-20:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _horizontal_basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(up[0])) >= 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = _normalize(np.cross(seed, up))
    forward = np.cross(up, right).astype(np.float32)
    return right, forward


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    out = np.zeros((4, 4), dtype=np.float32)
    out[0, 0] = f / max(aspect, 1e-8)
    out[1, 1] = f
    out[2, 2] = (far + near) / (near - far)
    out[2, 3] = (2.0 * far * near) / (near - far)
    out[3, 2] = -1.0
    return out


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(target - eye)
    s = _normalize(np.cross(f, up))
    u = np.cross(s, f).astype(np.float32)

    out = np.eye(4, dtype=np.float32)
    out[0, :3] = s
    out[1, :3] = u
    out[2, :3] = -f
    out[0, 3] = -float(np.dot(s, eye))
    out[1, 3] = -float(np.dot(u, eye))
    out[2, 3] = float(np.dot(f, eye))
    return out


@dataclass
class OrbitCamera:
    """Small orbit camera matching the native and web viewers."""

    width: int = 800
    height: int = 800
    world_up: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    distance: float = 6.0
    yaw: float = 0.0
    pitch: float = 0.35
    fov_y_deg: float = 40.0
    near_clip: float = 4.0
    far_clip: float = 100.0

    _dragging: bool = False
    _panning: bool = False
    _drag_x: float = 0.0
    _drag_y: float = 0.0
    _drag_yaw: float = 0.0
    _drag_pitch: float = 0.0
    _drag_target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    _view: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    _proj: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    _mvp: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    _eye: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    orbit_speed: float = 0.005
    pan_speed: float = 0.003
    scroll_step: float = 0.1

    def __post_init__(self) -> None:
        self.world_up = _normalize(self.world_up)
        self.update()

    @property
    def view(self) -> np.ndarray:
        return self._view

    @property
    def proj(self) -> np.ndarray:
        return self._proj

    @property
    def mvp(self) -> np.ndarray:
        return self._mvp

    @property
    def eye(self) -> np.ndarray:
        return self._eye

    def begin_drag(self, x: float, y: float, pan: bool) -> None:
        self._dragging = True
        self._panning = pan
        self._drag_x = x
        self._drag_y = y
        self._drag_yaw = self.yaw
        self._drag_pitch = self.pitch
        self._drag_target = self.target.copy()

    def drag_update(self, x: float, y: float) -> None:
        if not self._dragging:
            return

        dx = x - self._drag_x
        dy = y - self._drag_y
        if self._panning:
            speed = self.pan_speed * self.distance
            right = self._view[0, :3]
            up = self._view[1, :3]
            self.target = (
                self._drag_target
                - right * (dx * speed)
                + up * (dy * speed)
            ).astype(np.float32)
        else:
            self.yaw = self._drag_yaw + dx * self.orbit_speed
            self.pitch = self._drag_pitch + dy * self.orbit_speed
            limit = math.radians(89.0)
            self.pitch = max(-limit, min(limit, self.pitch))
        self.update()

    def end_drag(self) -> None:
        self._dragging = False

    def scroll(self, delta: float) -> None:
        self.distance *= (1.0 - self.scroll_step) if delta > 0.0 else (1.0 + self.scroll_step)
        self.distance = max(self.distance, 0.01)
        self.update()

    def frame_bounds(self, center: np.ndarray, radius: float) -> None:
        self.target = np.asarray(center, dtype=np.float32).copy()
        radius = max(radius, 1e-4)
        self.distance = radius * 1.6 / math.tan(math.radians(self.fov_y_deg * 0.5))
        self.yaw = 0.0
        self.pitch = 0.35
        self.update()

    def update(self) -> None:
        right, forward = _horizontal_basis(self.world_up)
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        offset = self.distance * (cp * (cy * forward + sy * right) + sp * self.world_up)
        self._eye = (self.target + offset).astype(np.float32)
        self._view = _look_at(self._eye, self.target, self.world_up)
        aspect = float(self.width) / float(self.height) if self.height > 0 else 1.0
        self._proj = _perspective(self.fov_y_deg, aspect, self.near_clip, self.far_clip)
        self._mvp = (self._proj @ self._view).astype(np.float32)
