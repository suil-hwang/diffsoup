"""NumPy orbit camera matching the native GLFW/GLM viewer controls."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / length


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Build a conventional right-handed OpenGL projection matrix."""
    if aspect <= 0 or near <= 0 or far <= near:
        raise ValueError("invalid perspective parameters")
    focal = 1.0 / np.tan(np.deg2rad(fov_y_deg) * 0.5)
    return np.array(
        [
            [focal / aspect, 0.0, 0.0, 0.0],
            [0.0, focal, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a GLM-compatible right-handed view matrix."""
    forward = _normalize(target - eye)
    side = _normalize(np.cross(forward, up))
    camera_up = np.cross(side, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = side
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(side, eye))
    matrix[1, 3] = -float(np.dot(camera_up, eye))
    matrix[2, 3] = float(np.dot(forward, eye))
    return matrix


@dataclass
class OrbitCamera:
    """Orbit, pan, and dolly camera with an arbitrary world-up vector."""

    width: int = 1200
    height: int = 1200
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

    def __post_init__(self) -> None:
        self.world_up = _normalize(np.asarray(self.world_up, dtype=np.float32))
        self.target = np.asarray(self.target, dtype=np.float32).copy()
        self._dragging = False
        self._panning = False
        self._drag_xy = np.zeros(2, dtype=np.float32)
        self._drag_angles = np.zeros(2, dtype=np.float32)
        self._drag_target = self.target.copy()
        self.eye = np.zeros(3, dtype=np.float32)
        self.view = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.mvp = np.eye(4, dtype=np.float32)
        self.update()

    def _horizontal_basis(self) -> tuple[np.ndarray, np.ndarray]:
        seed = (
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(self.world_up[0])) < 0.9
            else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        right = _normalize(np.cross(seed, self.world_up))
        forward = np.cross(self.world_up, right)
        return right, forward

    def update(self) -> None:
        right, forward = self._horizontal_basis()
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        offset = self.distance * (
            cp * (cy * forward + sy * right) + sp * self.world_up
        )
        self.eye = np.asarray(self.target + offset, dtype=np.float32)
        self.view = look_at(self.eye, self.target, self.world_up)
        aspect = self.width / self.height if self.height > 0 else 1.0
        self.projection = perspective(
            self.fov_y_deg, aspect, self.near_clip, self.far_clip
        )
        self.mvp = np.ascontiguousarray(self.projection @ self.view, dtype=np.float32)

    def begin_drag(self, x: float, y: float, pan: bool) -> None:
        self._dragging = True
        self._panning = pan
        self._drag_xy[:] = (x, y)
        self._drag_angles[:] = (self.yaw, self.pitch)
        self._drag_target = self.target.copy()

    def drag_update(self, x: float, y: float) -> None:
        if not self._dragging:
            return
        dx, dy = np.asarray((x, y), dtype=np.float32) - self._drag_xy
        if self._panning:
            speed = 0.003 * self.distance
            screen_right = self.view[0, :3]
            screen_up = self.view[1, :3]
            self.target = (
                self._drag_target - screen_right * dx * speed + screen_up * dy * speed
            )
        else:
            self.yaw = float(self._drag_angles[0] + dx * 0.005)
            limit = np.deg2rad(89.0)
            self.pitch = float(np.clip(self._drag_angles[1] + dy * 0.005, -limit, limit))
        self.update()

    def end_drag(self) -> None:
        self._dragging = False

    def scroll(self, delta: float) -> None:
        self.distance *= 0.9 if delta > 0 else 1.1
        self.distance = max(self.distance, 0.01)
        self.update()

