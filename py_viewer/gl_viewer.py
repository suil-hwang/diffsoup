"""PyQt5/PyOpenGL implementation of the DiffSoup viewer."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from pathlib import Path
import time
from typing import Any, cast

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .assets import SceneAssets
from .camera import OrbitCamera
from . import shaders

GL = cast(Any, import_module("OpenGL.GL"))


def _check_gl(label: str) -> None:
    err = GL.glGetError()
    if err != GL.GL_NO_ERROR:
        raise RuntimeError(f"OpenGL error after {label}: 0x{err:04x}")


def _compile_shader(shader_type: Any, source: str) -> int:
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    ok = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not ok:
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        GL.glDeleteShader(shader)
        raise RuntimeError(f"Shader compile failed:\n{log}")
    return int(shader)


def _link_program(vertex_source: str, fragment_source: str) -> int:
    vs = _compile_shader(GL.GL_VERTEX_SHADER, vertex_source)
    fs = _compile_shader(GL.GL_FRAGMENT_SHADER, fragment_source)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    ok = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    if not ok:
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        GL.glDeleteProgram(program)
        raise RuntimeError(f"Program link failed:\n{log}")
    return int(program)


def _float32_contiguous(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.float32 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


def _uniform_mat4(location: int, mat: np.ndarray) -> None:
    GL.glUniformMatrix4fv(
        location,
        1,
        GL.GL_TRUE,
        _float32_contiguous(mat),
    )


def _tile_weights_std140(
    weights: np.ndarray,
    out_dim: int,
    in_dim: int,
) -> np.ndarray:
    """Tile row-major weights into column-major mat4 blocks for std140."""

    src = np.asarray(weights, dtype=np.float32).reshape(out_dim, in_dim)
    dst = np.zeros(16 * 16, dtype=np.float32)
    for tr in range(4):
        for tc in range(4):
            tile = tr * 4 + tc
            for c in range(4):
                for r in range(4):
                    gr = tr * 4 + r
                    gc = tc * 4 + c
                    if gr < out_dim and gc < in_dim:
                        dst[tile * 16 + c * 4 + r] = src[gr, gc]
    return np.ascontiguousarray(dst)


def _tile_w3_std140(weights: np.ndarray) -> np.ndarray:
    """Tile the 3x16 output layer into four std140 mat4 values."""

    src = np.asarray(weights, dtype=np.float32).reshape(3, 16)
    dst = np.zeros(4 * 16, dtype=np.float32)
    for tc in range(4):
        for c in range(4):
            for r in range(3):
                gc = tc * 4 + c
                if gc < 16:
                    dst[tc * 16 + c * 4 + r] = src[r, gc]
    return np.ascontiguousarray(dst)


def _pad_vec4_std140(values: np.ndarray) -> np.ndarray:
    src = np.asarray(values, dtype=np.float32).reshape(-1)
    if src.size > 4:
        raise ValueError(f"Expected at most 4 values, got {src.size}")
    dst = np.zeros(4, dtype=np.float32)
    dst[: src.size] = src
    return np.ascontiguousarray(dst)


_UBO_BINDINGS = {
    "W1Block": 0,
    "B1Block": 1,
    "W2Block": 2,
    "B2Block": 3,
    "W3Block": 4,
    "B3Block": 5,
}


@dataclass(frozen=True)
class FinalRenderTarget:
    fbo: int
    tex: int
    width: int
    height: int


class DiffSoupGLWidget(QtWidgets.QOpenGLWidget):
    """Interactive GPU viewer for one exported DiffSoup scene."""

    fpsChanged = QtCore.pyqtSignal(float)
    resolutionChanged = QtCore.pyqtSignal(int, int)

    def __init__(
        self,
        scene: SceneAssets,
        output_dir: str | Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.scene_assets = scene
        self.output_dir = Path(output_dir) if output_dir else scene.scene_dir / "py_viewer_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.camera = OrbitCamera(world_up=scene.up)
        self.background = np.asarray(scene.background, dtype=np.float32)
        self._clear_color_a = np.empty(4, dtype=np.float32)
        self._clear_color_b = np.empty(4, dtype=np.float32)
        self._sync_clear_colors()
        self._frame_scene()

        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)

        self._geom_program = 0
        self._post_program = 0
        self._mesh_vao = 0
        self._mesh_vbo = 0
        self._tri_id_vbo = 0
        self._post_vao = 0
        self._lut_tex = [0, 0]
        self._fbo = 0
        self._color_tex = [0, 0]
        self._depth_tex = 0
        self._vertex_count = 0
        self._ubo = {
            "W1Block": 0,
            "B1Block": 0,
            "W2Block": 0,
            "B2Block": 0,
            "W3Block": 0,
            "B3Block": 0,
        }
        self._gl_cleaned = False
        self._cleanup_connected = False

        self._loc = {}
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(0)

        self._fps_t0 = time.perf_counter()
        self._fps_frames = 0

    def _frame_scene(self) -> None:
        verts = self.scene_assets.verts
        if verts.size == 0:
            return
        center = verts.mean(axis=0)
        self.camera.target = np.asarray(center, dtype=np.float32).copy()
        self.camera.distance = 6.0
        self.camera.yaw = 0.0
        self.camera.pitch = 0.35
        self.camera.update()

    def initializeGL(self) -> None:
        version = GL.glGetString(GL.GL_VERSION)
        if version:
            version_text = version.decode("ascii", errors="replace")
            print(f"[py_viewer] OpenGL {version_text}")
        context = self.context()
        if context is not None and not self._cleanup_connected:
            context.aboutToBeDestroyed.connect(self.cleanup_gl)
            self._cleanup_connected = True

        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_BLEND)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        self._geom_program = _link_program(shaders.GEOM_VS, shaders.GEOM_FS)
        self._post_program = _link_program(shaders.POST_VS, shaders.POST_FS)
        self._loc = {
            "geom_mvp": GL.glGetUniformLocation(self._geom_program, "uMVP"),
            "tri_tex_size": GL.glGetUniformLocation(self._geom_program, "uTriTexSize"),
            "tri_tex0": GL.glGetUniformLocation(self._geom_program, "uTriTex0"),
            "tri_tex1": GL.glGetUniformLocation(self._geom_program, "uTriTex1"),
            "level": GL.glGetUniformLocation(self._geom_program, "uLevel"),
            "post_tex_a": GL.glGetUniformLocation(self._post_program, "texA"),
            "post_tex_b": GL.glGetUniformLocation(self._post_program, "texB"),
            "inv_mvp": GL.glGetUniformLocation(self._post_program, "uInvMVP"),
        }

        self._post_vao = int(GL.glGenVertexArrays(1))
        self._init_mlp_ubos()
        self._upload_mesh()
        self._upload_luts()
        self._upload_mlp()
        self._resize_offscreen(max(self.width(), 1), max(self.height(), 1))
        _check_gl("initializeGL")

    def resizeGL(self, width: int, height: int) -> None:
        self.camera.width = max(width, 1)
        self.camera.height = max(height, 1)
        self.camera.update()
        self.resolutionChanged.emit(self.camera.width, self.camera.height)
        if self._geom_program:
            self._resize_offscreen(self.camera.width, self.camera.height)

    def paintGL(self) -> None:
        if not self._fbo:
            return
        self.camera.update()
        self._render(self.camera.mvp)
        self._update_fps()

    @property
    def is_ready(self) -> bool:
        return bool(self._fbo)

    def set_auto_update(self, enabled: bool) -> None:
        if enabled and not self._timer.isActive():
            self._timer.start(0)
        elif not enabled and self._timer.isActive():
            self._timer.stop()

    def set_render_size(self, width: int, height: int) -> None:
        self.makeCurrent()
        self.camera.width = max(width, 1)
        self.camera.height = max(height, 1)
        self.camera.update()
        if self._geom_program:
            self._resize_offscreen(self.camera.width, self.camera.height)
        self.resolutionChanged.emit(self.camera.width, self.camera.height)

    def create_final_render_target(self, width: int, height: int) -> FinalRenderTarget:
        self.makeCurrent()
        width = max(width, 1)
        height = max(height, 1)

        fbo = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            tex,
            0,
        )
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Final benchmark FBO incomplete: 0x{status:04x}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        return FinalRenderTarget(fbo=fbo, tex=tex, width=width, height=height)

    def delete_final_render_target(self, target: FinalRenderTarget) -> None:
        self.makeCurrent()
        if target.tex:
            GL.glDeleteTextures(1, [target.tex])
        if target.fbo:
            GL.glDeleteFramebuffers(1, [target.fbo])
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def render_mvp_once(
        self,
        mvp: np.ndarray,
        *,
        inv_mvp: np.ndarray | None = None,
        target_fbo: int | None = None,
        blit_to_default: bool = False,
    ) -> None:
        if not self._fbo:
            raise RuntimeError("OpenGL viewer is not initialized.")
        self.makeCurrent()
        self._render(
            _float32_contiguous(mvp),
            inv_mvp=None if inv_mvp is None else _float32_contiguous(inv_mvp),
            target_fbo=target_fbo,
            blit_to_default=blit_to_default,
        )

    def read_current_rgba(
        self,
        *,
        source_fbo: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        width = max(self.camera.width if width is None else width, 1)
        height = max(self.camera.height if height is None else height, 1)
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        self.makeCurrent()
        read_fbo = self.defaultFramebufferObject() if source_fbo is None else source_fbo
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, read_fbo)
        GL.glReadPixels(
            0,
            0,
            width,
            height,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            pixels,
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        return np.flipud(pixels).copy()

    @staticmethod
    def _nonzero_ids(values: list[int]) -> list[int]:
        return [value for value in values if value]

    def cleanup_gl(self) -> None:
        if self._gl_cleaned:
            return
        self._gl_cleaned = True

        try:
            self.makeCurrent()
        except Exception:
            pass

        for attr in ("_geom_program", "_post_program"):
            program = int(getattr(self, attr))
            if program:
                GL.glDeleteProgram(program)
                setattr(self, attr, 0)

        vaos = self._nonzero_ids([self._mesh_vao, self._post_vao])
        if vaos:
            GL.glDeleteVertexArrays(len(vaos), vaos)
        self._mesh_vao = 0
        self._post_vao = 0

        buffer_ids = [self._mesh_vbo, self._tri_id_vbo]
        buffer_ids.extend(self._ubo[name] for name in _UBO_BINDINGS)
        buffers = self._nonzero_ids(buffer_ids)
        if buffers:
            GL.glDeleteBuffers(len(buffers), buffers)
        self._mesh_vbo = 0
        self._tri_id_vbo = 0
        self._ubo = {name: 0 for name in _UBO_BINDINGS}

        textures = self._nonzero_ids(
            [*self._lut_tex, *self._color_tex, self._depth_tex]
        )
        if textures:
            GL.glDeleteTextures(len(textures), textures)
        self._lut_tex = [0, 0]
        self._color_tex = [0, 0]
        self._depth_tex = 0

        fbos = self._nonzero_ids([self._fbo])
        if fbos:
            GL.glDeleteFramebuffers(len(fbos), fbos)
        self._fbo = 0
        self._vertex_count = 0
        self._loc = {}

        try:
            self.doneCurrent()
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cleanup_gl()
        super().closeEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        pan = (
            event.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton)
            or bool(event.modifiers() & QtCore.Qt.ShiftModifier)
        )
        self.camera.begin_drag(float(event.x()), float(event.y()), pan)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self.camera.drag_update(float(event.x()), float(event.y()))
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.camera.end_drag()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self.camera.scroll(float(event.angleDelta().y()))
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key_R:
            self.reset_camera()
        elif key == QtCore.Qt.Key_S:
            self.save_screenshot()
        elif key == QtCore.Qt.Key_Escape:
            self.window().close()
        else:
            super().keyPressEvent(event)

    def reset_camera(self) -> None:
        self._frame_scene()
        self.update()

    def save_screenshot(self) -> Path:
        path = self.output_dir / "screenshot.png"
        self.grabFramebuffer().save(str(path))
        print(f"[py_viewer] saved {path}")
        return path

    def set_background(self, rgb: tuple[float, float, float] | np.ndarray) -> None:
        self.background = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
        self._sync_clear_colors()
        self.update()

    def _sync_clear_colors(self) -> None:
        self._clear_color_a[:3] = self.background[:3]
        self._clear_color_a[3] = 1.0
        self._clear_color_b[:3] = self.background[:3]
        self._clear_color_b[3] = 0.0

    def set_fov_y(self, value: float) -> None:
        self.camera.fov_y_deg = value
        self.camera.update()
        self.update()

    def set_near_clip(self, value: float) -> None:
        self.camera.near_clip = max(value, 1e-4)
        if self.camera.far_clip <= self.camera.near_clip:
            self.camera.far_clip = self.camera.near_clip + 1e-3
        self.camera.update()
        self.update()

    def set_far_clip(self, value: float) -> None:
        self.camera.far_clip = max(value, self.camera.near_clip + 1e-3)
        self.camera.update()
        self.update()

    def _upload_mesh(self) -> None:
        verts = self.scene_assets.verts
        faces = self.scene_assets.faces
        valid = np.all((faces >= 0) & (faces < verts.shape[0]), axis=1)
        valid_faces = faces[valid]
        valid_ids = np.nonzero(valid)[0].astype(np.uint32)

        tri_positions = np.ascontiguousarray(verts[valid_faces.reshape(-1)], dtype=np.float32)
        tri_ids = np.ascontiguousarray(np.repeat(valid_ids, 3), dtype=np.uint32)
        self._vertex_count = int(tri_positions.shape[0])

        self._mesh_vao = int(GL.glGenVertexArrays(1))
        self._mesh_vbo = int(GL.glGenBuffers(1))
        self._tri_id_vbo = int(GL.glGenBuffers(1))

        GL.glBindVertexArray(self._mesh_vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._mesh_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, tri_positions.nbytes, tri_positions, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._tri_id_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, tri_ids.nbytes, tri_ids, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribIPointer(1, 1, GL.GL_UNSIGNED_INT, 0, None)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _upload_luts(self) -> None:
        max_texture = GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE)
        for idx, image in enumerate((self.scene_assets.lut0, self.scene_assets.lut1)):
            height, width, _ = image.shape
            if width > max_texture or height > max_texture:
                raise RuntimeError(
                    f"LUT texture {idx} is {width}x{height}, larger than GL max {max_texture}"
                )
            tex = int(GL.glGenTextures(1))
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA8,
                int(width),
                int(height),
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                np.ascontiguousarray(image),
            )
            self._lut_tex[idx] = tex
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _init_mlp_ubos(self) -> None:
        sizes = {
            "W1Block": 16 * 16 * 4,
            "B1Block": 16 * 4,
            "W2Block": 16 * 16 * 4,
            "B2Block": 16 * 4,
            "W3Block": 4 * 16 * 4,
            "B3Block": 4 * 4,
        }
        for block_name, binding in _UBO_BINDINGS.items():
            block_index = GL.glGetUniformBlockIndex(self._post_program, block_name)
            invalid_index = cast(int, GL.GL_INVALID_INDEX)
            if int(block_index) == invalid_index:
                raise RuntimeError(f"Uniform block not found: {block_name}")
            GL.glUniformBlockBinding(self._post_program, block_index, binding)
            ubo = int(GL.glGenBuffers(1))
            GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, ubo)
            GL.glBufferData(
                GL.GL_UNIFORM_BUFFER,
                sizes[block_name],
                None,
                GL.GL_DYNAMIC_DRAW,
            )
            GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, binding, ubo)
            self._ubo[block_name] = ubo
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def _upload_ubo(self, block_name: str, data: np.ndarray) -> None:
        arr = np.ascontiguousarray(data, dtype=np.float32)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self._ubo[block_name])
        GL.glBufferSubData(GL.GL_UNIFORM_BUFFER, 0, arr.nbytes, arr)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def _upload_mlp(self) -> None:
        scene = self.scene_assets
        self._upload_ubo("W1Block", _tile_weights_std140(scene.W1, 16, 16))
        self._upload_ubo("B1Block", np.ascontiguousarray(scene.b1, dtype=np.float32))
        self._upload_ubo("W2Block", _tile_weights_std140(scene.W2, 16, 16))
        self._upload_ubo("B2Block", np.ascontiguousarray(scene.b2, dtype=np.float32))
        self._upload_ubo("W3Block", _tile_w3_std140(scene.W3))
        self._upload_ubo("B3Block", _pad_vec4_std140(scene.b3))

    def _resize_offscreen(self, width: int, height: int) -> None:
        if self._fbo:
            GL.glDeleteFramebuffers(1, [self._fbo])
            GL.glDeleteTextures(2, self._color_tex)
            GL.glDeleteTextures(1, [self._depth_tex])
            self._fbo = 0
            self._color_tex = [0, 0]
            self._depth_tex = 0

        self._fbo = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)

        self._color_tex = [int(GL.glGenTextures(1)), int(GL.glGenTextures(1))]
        for i, tex in enumerate(self._color_tex):
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA8,
                width,
                height,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                None,
            )
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                cast(int, GL.GL_COLOR_ATTACHMENT0) + i,
                GL.GL_TEXTURE_2D,
                tex,
                0,
            )

        self._depth_tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._depth_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT24,
            width,
            height,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_UNSIGNED_INT,
            None,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self._depth_tex,
            0,
        )

        draw_buffers = np.array(
            [GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1],
            dtype=np.uint32,
        )
        GL.glDrawBuffers(2, draw_buffers)
        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Offscreen FBO incomplete: 0x{status:04x}")
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def _render(
        self,
        mvp: np.ndarray,
        *,
        inv_mvp: np.ndarray | None = None,
        target_fbo: int | None = None,
        blit_to_default: bool = False,
    ) -> None:
        width = max(self.camera.width, 1)
        height = max(self.camera.height, 1)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, width, height)
        GL.glClearBufferfv(GL.GL_COLOR, 0, self._clear_color_a)
        GL.glClearBufferfv(GL.GL_COLOR, 1, self._clear_color_b)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glUseProgram(self._geom_program)
        _uniform_mat4(self._loc["geom_mvp"], mvp)
        GL.glUniform1i(self._loc["tri_tex0"], 0)
        GL.glUniform1i(self._loc["tri_tex1"], 1)
        GL.glUniform2i(self._loc["tri_tex_size"], int(self.scene_assets.lut0.shape[1]), int(self.scene_assets.lut0.shape[0]))
        GL.glUniform1i(self._loc["level"], self.scene_assets.level)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._lut_tex[0])
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._lut_tex[1])
        GL.glBindVertexArray(self._mesh_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._vertex_count)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        final_fbo = self.defaultFramebufferObject() if target_fbo is None else target_fbo
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, final_fbo)
        GL.glViewport(0, 0, width, height)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self._post_program)
        GL.glUniform1i(self._loc["post_tex_a"], 0)
        GL.glUniform1i(self._loc["post_tex_b"], 1)
        if inv_mvp is None:
            inv_mvp = np.linalg.inv(mvp).astype(np.float32)
        _uniform_mat4(self._loc["inv_mvp"], inv_mvp)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex[0])
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_tex[1])
        GL.glBindVertexArray(self._post_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        if target_fbo is not None and blit_to_default:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, final_fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())
            GL.glBlitFramebuffer(
                0,
                0,
                width,
                height,
                0,
                0,
                width,
                height,
                GL.GL_COLOR_BUFFER_BIT,
                GL.GL_NEAREST,
            )
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())

    def _update_fps(self) -> None:
        self._fps_frames += 1
        now = time.perf_counter()
        elapsed = now - self._fps_t0
        if elapsed >= 0.5:
            fps = self._fps_frames / elapsed
            self._fps_t0 = now
            self._fps_frames = 0
            self.fpsChanged.emit(fps)


class ViewerControls(QtWidgets.QWidget):
    """Settings panel for the interactive viewer."""

    FOV_MIN = 10.0
    FOV_MAX = 120.0
    NEAR_MIN = 0.01
    NEAR_MAX = 50.0
    FAR_MIN = 1.0
    FAR_MAX = 1000.0
    LOG_SLIDER_STEPS = 1000
    PANEL_SIZE = QtCore.QSize(280, 260)
    PANEL_OFFSET = QtCore.QPoint(20, 20)

    def __init__(self, viewer: DiffSoupGLWidget, parent=None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._syncing = False

        self.resolution_label = QtWidgets.QLabel()
        self.faces_label = QtWidgets.QLabel(
            f"Faces: {viewer.scene_assets.faces.shape[0]:,}"
        )

        self.background_button = QtWidgets.QPushButton()
        self.background_button.setFixedSize(32, 22)
        self.background_button.clicked.connect(self._choose_background)

        self.bg_r_spin = self._make_spin(0.0, 1.0, 0.01, 3, "")
        self.bg_g_spin = self._make_spin(0.0, 1.0, 0.01, 3, "")
        self.bg_b_spin = self._make_spin(0.0, 1.0, 0.01, 3, "")
        for spin in (self.bg_r_spin, self.bg_g_spin, self.bg_b_spin):
            spin.setFixedWidth(52)
            spin.valueChanged.connect(self._on_background_spin_changed)

        self.fov_slider = self._make_slider(
            int(self.FOV_MIN),
            int(self.FOV_MAX),
            page_step=5,
        )
        self.near_slider = self._make_slider(
            0,
            self.LOG_SLIDER_STEPS,
            page_step=50,
        )
        self.far_slider = self._make_slider(
            0,
            self.LOG_SLIDER_STEPS,
            page_step=50,
        )

        self.fov_spin = self._make_spin(self.FOV_MIN, self.FOV_MAX, 1.0, 1, " deg")
        self.near_spin = self._make_spin(self.NEAR_MIN, self.NEAR_MAX, 0.01, 3, "")
        self.far_spin = self._make_spin(self.FAR_MIN, self.FAR_MAX, 1.0, 1, "")
        for spin in (self.fov_spin, self.near_spin, self.far_spin):
            spin.setFixedWidth(72)

        self.fov_slider.valueChanged.connect(lambda value: self.set_fov_value(float(value)))
        self.near_slider.valueChanged.connect(
            lambda value: self.set_near_value(
                self._log_value_from_slider(value, self.NEAR_MIN, self.NEAR_MAX)
            )
        )
        self.far_slider.valueChanged.connect(
            lambda value: self.set_far_value(
                self._log_value_from_slider(value, self.FAR_MIN, self.FAR_MAX)
            )
        )

        self.fov_spin.valueChanged.connect(lambda value: self.set_fov_value(float(value)))
        self.near_spin.valueChanged.connect(lambda value: self.set_near_value(float(value)))
        self.far_spin.valueChanged.connect(lambda value: self.set_far_value(float(value)))

        self.reset_button = QtWidgets.QPushButton("Reset view")
        self.reset_button.clicked.connect(self.viewer.reset_camera)

        self.screenshot_button = QtWidgets.QPushButton("Save screenshot")
        self.screenshot_button.clicked.connect(self.viewer.save_screenshot)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("Background", self._background_row())
        form.addRow("FOV", self._slider_row(self.fov_slider, self.fov_spin))
        form.addRow("Near clip", self._slider_row(self.near_slider, self.near_spin))
        form.addRow("Far clip", self._slider_row(self.far_slider, self.far_spin))

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.screenshot_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.resolution_label)
        layout.addWidget(self.faces_label)
        layout.addSpacing(8)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.update_resolution(viewer.camera.width, viewer.camera.height)
        bg = viewer.background
        self.set_background_rgb((float(bg[0]), float(bg[1]), float(bg[2])))
        self.set_fov_value(viewer.camera.fov_y_deg)
        self.set_near_value(viewer.camera.near_clip)
        self.set_far_value(viewer.camera.far_clip)

    def _make_spin(
        self,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        suffix: str,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    def _make_slider(
        self,
        minimum: int,
        maximum: int,
        page_step: int,
    ) -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setSingleStep(1)
        slider.setPageStep(page_step)
        slider.setMinimumWidth(76)
        return slider

    def _background_row(self) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.background_button)
        layout.addWidget(self.bg_r_spin)
        layout.addWidget(self.bg_g_spin)
        layout.addWidget(self.bg_b_spin)
        return row

    def _slider_row(
        self,
        slider: QtWidgets.QSlider,
        spin: QtWidgets.QDoubleSpinBox,
    ) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        return row

    def _set_numeric_widget_value(
        self,
        widget: QtWidgets.QAbstractSpinBox | QtWidgets.QSlider,
        value: float | int,
    ) -> None:
        was_blocked = widget.blockSignals(True)
        if isinstance(widget, QtWidgets.QSlider):
            widget.setValue(round(value))
        else:
            widget.setValue(value)
        widget.blockSignals(was_blocked)

    def _sync_linear_value(
        self,
        spin: QtWidgets.QDoubleSpinBox,
        slider: QtWidgets.QSlider,
        value: float,
    ) -> None:
        self._set_numeric_widget_value(spin, value)
        self._set_numeric_widget_value(slider, value)

    def _sync_log_value(
        self,
        spin: QtWidgets.QDoubleSpinBox,
        slider: QtWidgets.QSlider,
        value: float,
        minimum: float,
        maximum: float,
    ) -> None:
        self._set_numeric_widget_value(spin, value)
        self._set_numeric_widget_value(
            slider,
            self._slider_from_log_value(value, minimum, maximum),
        )

    @classmethod
    def _slider_from_log_value(
        cls,
        value: float,
        minimum: float,
        maximum: float,
    ) -> int:
        clipped = min(max(value, minimum), maximum)
        denom = math.log(maximum) - math.log(minimum)
        t = (math.log(clipped) - math.log(minimum)) / denom
        return round(t * cls.LOG_SLIDER_STEPS)

    @classmethod
    def _log_value_from_slider(
        cls,
        slider_value: int,
        minimum: float,
        maximum: float,
    ) -> float:
        t = min(max(float(slider_value) / cls.LOG_SLIDER_STEPS, 0.0), 1.0)
        return math.exp(math.log(minimum) + t * (math.log(maximum) - math.log(minimum)))

    def update_resolution(self, width: int, height: int) -> None:
        self.resolution_label.setText(f"Resolution: {width}x{height}")

    def set_background_rgb(self, rgb: tuple[float, float, float]) -> None:
        clamped = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
        self.viewer.set_background(clamped)
        for spin, value in zip(
            (self.bg_r_spin, self.bg_g_spin, self.bg_b_spin),
            clamped,
        ):
            self._set_numeric_widget_value(spin, float(value))
        r, g, b = (round(c * 255.0) for c in clamped)
        self.background_button.setText("")
        self.background_button.setStyleSheet(
            "QPushButton { "
            f"background-color: rgb({r}, {g}, {b}); "
            "border: 1px solid #666; "
            "}"
        )

    def _on_background_spin_changed(self) -> None:
        self.set_background_rgb(
            (
                float(self.bg_r_spin.value()),
                float(self.bg_g_spin.value()),
                float(self.bg_b_spin.value()),
            )
        )

    def set_fov_value(self, value: float) -> None:
        fov = min(max(value, self.FOV_MIN), self.FOV_MAX)
        self.viewer.set_fov_y(fov)
        self._sync_linear_value(self.fov_spin, self.fov_slider, fov)

    def set_near_value(self, value: float) -> None:
        near = min(max(value, self.NEAR_MIN), self.NEAR_MAX)
        self.viewer.set_near_clip(near)
        self._sync_log_value(
            self.near_spin,
            self.near_slider,
            self.viewer.camera.near_clip,
            self.NEAR_MIN,
            self.NEAR_MAX,
        )
        self._sync_log_value(
            self.far_spin,
            self.far_slider,
            self.viewer.camera.far_clip,
            self.FAR_MIN,
            self.FAR_MAX,
        )

    def set_far_value(self, value: float) -> None:
        far = min(max(value, self.FAR_MIN), self.FAR_MAX)
        self.viewer.set_far_clip(far)
        self._sync_log_value(
            self.far_spin,
            self.far_slider,
            self.viewer.camera.far_clip,
            self.FAR_MIN,
            self.FAR_MAX,
        )

    def _choose_background(self) -> None:
        rgb = np.clip(self.viewer.background, 0.0, 1.0)
        color = QtGui.QColor.fromRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        chosen = QtWidgets.QColorDialog.getColor(
            color,
            self,
            "Background",
            QtWidgets.QColorDialog.DontUseNativeDialog,
        )
        if chosen.isValid():
            self.set_background_rgb(
                (chosen.redF(), chosen.greenF(), chosen.blueF())
            )


class DiffSoupViewerWindow(QtWidgets.QMainWindow):
    """Small window wrapper around ``DiffSoupGLWidget``."""

    def __init__(
        self,
        scene: SceneAssets,
        output_dir: str | Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.viewer = DiffSoupGLWidget(scene, output_dir=output_dir, parent=self)
        self.setCentralWidget(self.viewer)
        self._last_fps: float | None = None

        self.controls = ViewerControls(self.viewer, parent=self)
        self.controls_dock = QtWidgets.QDockWidget("Settings", self)
        self.controls_dock.setObjectName("settingsDock")
        self.controls_dock.setWidget(self.controls)
        self.controls_dock.setFloating(True)
        self.controls_dock.resize(ViewerControls.PANEL_SIZE)
        self.controls_dock.setMinimumSize(ViewerControls.PANEL_SIZE)
        self.controls_dock.setMaximumSize(ViewerControls.PANEL_SIZE)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.controls_dock)
        QtCore.QTimer.singleShot(0, self._position_controls_dock)

        self.statusBar().showMessage(
            "Left drag: orbit | right/shift drag: pan | wheel: zoom | R: reset | S: screenshot | Esc: quit"
        )
        self.viewer.fpsChanged.connect(self._set_fps)
        self.viewer.resolutionChanged.connect(self._on_resolution_changed)
        self._update_title()

    def _set_fps(self, fps: float) -> None:
        self._last_fps = fps
        self._update_title()

    def _on_resolution_changed(self, width: int, height: int) -> None:
        self.controls.update_resolution(width, height)
        self._update_title()

    def _position_controls_dock(self) -> None:
        if not self.controls_dock.isFloating():
            return
        self.controls_dock.move(
            self.frameGeometry().topLeft() + ViewerControls.PANEL_OFFSET
        )

    def _update_title(self) -> None:
        width = self.viewer.camera.width
        height = self.viewer.camera.height
        fps_text = "" if self._last_fps is None else f"  FPS: {self._last_fps:.1f}"
        self.setWindowTitle(f"Viewer  {width}x{height}{fps_text}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.viewer.cleanup_gl()
        super().closeEvent(event)
