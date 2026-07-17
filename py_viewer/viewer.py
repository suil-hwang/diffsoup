# py_viewer/viewer.py

from __future__ import annotations

import importlib
import time
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .camera import OrbitCamera
from .scene import SceneData


class RenderMode(IntEnum):
    """Post-processing output selected by the viewer."""

    COLOR = 0
    DEPTH = 1
    NORMAL = 2

    @property
    def cli_name(self) -> str:
        return self.name.lower()

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @classmethod
    def coerce(cls, value: "RenderMode | str | int") -> "RenderMode":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError as exc:
                raise ValueError(f"unknown render mode: {value!r}") from exc
        try:
            return cls(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown render mode: {value!r}") from exc


class DepthRange(Enum):
    """Range used to map positive camera-space depth to display grayscale."""

    AUTO = "auto"
    CLIP = "clip"

    @classmethod
    def coerce(cls, value: "DepthRange | str") -> "DepthRange":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError as exc:
                raise ValueError(f"unknown depth range: {value!r}") from exc
        raise ValueError(f"unknown depth range: {value!r}")


class NormalOrientation(Enum):
    """Whether two-sided normals follow winding or face the current camera."""

    FACE_FORWARD = "face-forward"
    ORIENTED = "oriented"

    @classmethod
    def coerce(
        cls, value: "NormalOrientation | str"
    ) -> "NormalOrientation":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError as exc:
                raise ValueError(
                    f"unknown normal orientation: {value!r}"
                ) from exc
        raise ValueError(f"unknown normal orientation: {value!r}")


def _load_shader_source(name: str) -> str:
    path = Path(__file__).with_name(name)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read shader source: {path}") from exc
    if not source.lstrip().startswith("#version"):
        raise RuntimeError(f"Shader has no #version directive: {path}")
    return source


def _load_runtime(interactive: bool):
    try:
        glfw = importlib.import_module("glfw")
        moderngl = importlib.import_module("moderngl")
    except ImportError as exc:
        raise RuntimeError(
            "The Python viewer requires glfw and ModernGL. "
            "Install them with `pip install glfw moderngl`."
        ) from exc

    imgui = None
    glfw_renderer = None
    if interactive:
        try:
            imgui = importlib.import_module("imgui")
            integration = importlib.import_module("imgui.integrations.glfw")
        except ImportError as exc:
            raise RuntimeError(
                "Interactive mode requires pyimgui and its PyOpenGL backend. "
                "Install them with `pip install imgui PyOpenGL`."
            ) from exc
        glfw_renderer = integration.GlfwRenderer
    return glfw, moderngl, imgui, glfw_renderer


def _tile_weights(weights: np.ndarray) -> np.ndarray:
    """Split a 16x16 row-major matrix into sixteen 4x4 GLSL matrices."""
    tiles = np.zeros((16, 4, 4), dtype=np.float32)
    for tile_row in range(4):
        for tile_col in range(4):
            tiles[tile_row * 4 + tile_col] = weights[
                tile_row * 4 : tile_row * 4 + 4,
                tile_col * 4 : tile_col * 4 + 4,
            ]
    return np.ascontiguousarray(tiles)


def _tile_output_weights(weights: np.ndarray) -> np.ndarray:
    tiles = np.zeros((4, 4, 4), dtype=np.float32)
    for tile_col in range(4):
        tiles[tile_col, :3, :] = weights[:, tile_col * 4 : tile_col * 4 + 4]
    return np.ascontiguousarray(tiles)


def _matrix_bytes(matrix: np.ndarray) -> bytes:
    """Return one row-major matrix as column-major OpenGL bytes."""
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float32).T).tobytes()


def _matrix_array_bytes(matrices: np.ndarray) -> bytes:
    """Return row-major matrices as column-major OpenGL bytes."""
    matrices = np.asarray(matrices, dtype=np.float32)
    return np.ascontiguousarray(matrices.transpose(0, 2, 1)).tobytes()


def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return finite unit normals for an indexed triangle array."""
    triangles = np.asarray(verts, dtype=np.float32)[np.asarray(faces, dtype=np.int32)]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid] = (0.0, 0.0, 1.0)
    return np.ascontiguousarray(normals, dtype=np.float32)


def _linearize_window_depth(
    window_depth: np.ndarray,
    near_clip: float,
    far_clip: float,
) -> np.ndarray:
    """Convert conventional OpenGL window depth to positive camera-axis depth."""
    near_clip = float(near_clip)
    far_clip = float(far_clip)
    if near_clip <= 0.0 or far_clip <= near_clip:
        raise ValueError("depth clips must satisfy 0 < near < far")
    window_depth = np.asarray(window_depth, dtype=np.float64)
    ndc_depth = window_depth * 2.0 - 1.0
    denominator = far_clip + near_clip - ndc_depth * (far_clip - near_clip)
    return 2.0 * near_clip * far_clip / denominator


def _automatic_depth_range(
    verts: np.ndarray,
    mvp: np.ndarray,
    near_clip: float,
    far_clip: float,
) -> tuple[float, float]:
    """Estimate a robust visible range without reading the GPU depth buffer."""
    verts = np.asarray(verts, dtype=np.float64)
    mvp = np.asarray(mvp, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or mvp.shape != (4, 4):
        raise ValueError("expected verts [V, 3] and mvp [4, 4]")

    homogeneous = np.concatenate(
        [verts, np.ones((len(verts), 1), dtype=np.float64)], axis=1
    )
    clip = homogeneous @ mvp.T
    valid_w = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 1e-8)
    safe_w = np.where(valid_w, clip[:, 3], 1.0)
    ndc = clip[:, :3] / safe_w[:, None]
    in_clip = valid_w & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    in_view = (
        in_clip
        & (np.abs(ndc[:, 0]) <= 1.05)
        & (np.abs(ndc[:, 1]) <= 1.05)
    )
    selected = in_view if np.count_nonzero(in_view) >= 8 else in_clip
    if not np.any(selected):
        return float(near_clip), float(far_clip)

    window_depth = ndc[selected, 2] * 0.5 + 0.5
    depths = _linearize_window_depth(window_depth, near_clip, far_clip)
    depths = depths[np.isfinite(depths)]
    if not len(depths):
        return float(near_clip), float(far_clip)

    if len(depths) >= 100:
        display_near, display_far = np.percentile(depths, (1.0, 99.0))
    else:
        display_near, display_far = float(depths.min()), float(depths.max())
    span = float(display_far - display_near)
    minimum_span = max(1e-4 * (far_clip - near_clip), 1e-4)
    if span <= minimum_span:
        center = 0.5 * float(display_near + display_far)
        margin = max(0.05 * max(center, near_clip), 0.01 * (far_clip - near_clip))
        display_near, display_far = center - margin, center + margin
    else:
        padding = 0.02 * span
        display_near -= padding
        display_far += padding

    display_near = max(float(near_clip), float(display_near))
    display_far = min(float(far_clip), float(display_far))
    if display_far - display_near <= minimum_span:
        return float(near_clip), float(far_clip)
    return display_near, display_far


class Viewer:
    """Two-pass OpenGL viewer backed by Python-managed GPU resources."""

    def __init__(
        self,
        scene: SceneData,
        *,
        width: int = 1200,
        height: int = 1200,
        output_dir: str | Path = "./results/py_viewer",
        interactive: bool = True,
        render_mode: RenderMode | str | int = RenderMode.COLOR,
        depth_range: DepthRange | str = DepthRange.AUTO,
        normal_orientation: NormalOrientation | str = NormalOrientation.FACE_FORWARD,
    ) -> None:
        self.scene = scene
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        self.background = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.render_mode = RenderMode.coerce(render_mode)
        self.depth_range = DepthRange.coerce(depth_range)
        self.normal_orientation = NormalOrientation.coerce(normal_orientation)
        self.depth_display_near = 0.0
        self.depth_display_far = 1.0
        self._depth_range_cache_mvp: np.ndarray | None = None
        self._depth_range_cache_clips: tuple[float, float] | None = None

        self.glfw, self.moderngl, self.imgui, renderer_cls = _load_runtime(
            interactive
        )
        self.window = None
        self.imgui_renderer = None
        self.imgui_context = None
        self.ctx: Any = None
        self.geom_program: Any = None
        self.post_program: Any = None
        self.geom_vao: Any = None
        self.post_vao: Any = None
        self.position_buffer: Any = None
        self.triangle_id_buffer: Any = None
        self.normal_buffer: Any = None
        self.lut_textures: list[Any] = []
        self.geometry_fbo: Any = None
        self.color_textures: list[Any] = []
        self.depth_texture: Any = None
        self._attachment_clear_fbos: list[Any] = []
        self._closed = False

        self.camera = OrbitCamera(
            width=max(1, int(width)),
            height=max(1, int(height)),
            world_up=np.asarray(scene.up, dtype=np.float32),
            target=scene.center,
        )

        try:
            self._create_window(width, height, visible=interactive)
            if interactive:
                self.imgui_context = self.imgui.create_context()
                self.imgui.get_io().ini_file_name = None
                self.imgui_renderer = renderer_cls(
                    self.window, attach_callbacks=False
                )

            self._install_callbacks()
            self._init_gl()
        except Exception:
            self.close()
            raise

    def _create_window(self, width: int, height: int, *, visible: bool) -> None:
        glfw = self.glfw
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        self.window = glfw.create_window(
            int(width), int(height), "DiffSoup Python Viewer", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

    def _install_callbacks(self) -> None:
        glfw = self.glfw
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)

    def _init_gl(self) -> None:
        self.ctx = self.moderngl.create_context(require=410)
        self.ctx.gc_mode = None
        self.ctx.depth_func = "<"
        self.ctx.wireframe = False

        self.geom_program = self.ctx.program(
            vertex_shader=_load_shader_source("geometry.vert.glsl"),
            fragment_shader=_load_shader_source("geometry.frag.glsl"),
        )
        self.post_program = self.ctx.program(
            vertex_shader=_load_shader_source("post.vert.glsl"),
            fragment_shader=_load_shader_source("post.frag.glsl"),
        )

        self._upload_mesh()
        self.post_vao = self.ctx.vertex_array(self.post_program, [])
        self.lut_textures = []
        for rgba in (self.scene.lut0, self.scene.lut1):
            self.lut_textures.append(self._upload_texture(rgba))
        self.geom_program["uTriTex0"].value = 0
        self.geom_program["uTriTex1"].value = 1
        self.geom_program["uTriTexSize"].value = (
            int(self.scene.lut0.shape[1]),
            int(self.scene.lut0.shape[0]),
        )
        self.geom_program["uLevel"].value = self.scene.level
        self.post_program["texA"].value = 0
        self.post_program["texB"].value = 1
        self.post_program["texNormal"].value = 2
        self.post_program["texDepth"].value = 3
        self._upload_mlp()

        width, height = self.glfw.get_framebuffer_size(self.window)
        self._resize_fbo(max(1, width), max(1, height))

    def _upload_mesh(self) -> None:
        positions = np.ascontiguousarray(
            self.scene.verts[self.scene.faces].reshape(-1, 3), dtype=np.float32
        )
        triangle_ids = np.ascontiguousarray(
            np.repeat(np.arange(len(self.scene.faces), dtype=np.uint32), 3)
        )
        normals = np.ascontiguousarray(
            np.repeat(_face_normals(self.scene.verts, self.scene.faces), 3, axis=0)
        )
        self.vertex_count = int(len(positions))

        self.position_buffer = self.ctx.buffer(positions.tobytes())
        self.triangle_id_buffer = self.ctx.buffer(triangle_ids.tobytes())
        self.normal_buffer = self.ctx.buffer(normals.tobytes())
        self.geom_vao = self.ctx.vertex_array(
            self.geom_program,
            [
                (self.position_buffer, "3f", "aPos"),
                (self.triangle_id_buffer, "1u", "aTriID"),
                (self.normal_buffer, "3f", "aNormal"),
            ],
        )

    def _upload_texture(self, rgba: np.ndarray) -> Any:
        height, width, _ = rgba.shape
        texture = self.ctx.texture(
            (width, height),
            4,
            rgba.tobytes(),
            alignment=1,
            dtype="f1",
        )
        texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def _upload_mlp(self) -> None:
        self.post_program["W1"].write(
            _matrix_array_bytes(_tile_weights(self.scene.w1))
        )
        self.post_program["B1"].write(self.scene.b1.tobytes())
        self.post_program["W2"].write(
            _matrix_array_bytes(_tile_weights(self.scene.w2))
        )
        self.post_program["B2"].write(self.scene.b2.tobytes())
        self.post_program["W3"].write(
            _matrix_array_bytes(_tile_output_weights(self.scene.w3))
        )
        b3 = np.array([*self.scene.b3, 0.0], dtype=np.float32)
        self.post_program["B3"].write(b3.tobytes())

    def _release_geometry_targets(self) -> None:
        for framebuffer in reversed(self._attachment_clear_fbos):
            framebuffer.release()
        self._attachment_clear_fbos = []
        if self.geometry_fbo is not None:
            self.geometry_fbo.release()
            self.geometry_fbo = None
        if self.depth_texture is not None:
            self.depth_texture.release()
            self.depth_texture = None
        for texture in reversed(self.color_textures):
            texture.release()
        self.color_textures = []

    def _resize_fbo(self, width: int, height: int) -> None:
        width, height = max(1, int(width)), max(1, int(height))
        self.camera.width = width
        self.camera.height = height
        self.camera.update()

        self._release_geometry_targets()
        color_textures = [
            self.ctx.texture((width, height), 4, dtype="f1")
            for _ in range(3)
        ]
        depth_texture = self.ctx.depth_texture((width, height))
        clear_fbos: list[Any] = []
        geometry_fbo: Any = None
        try:
            for texture in color_textures:
                texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
                texture.repeat_x = False
                texture.repeat_y = False
            depth_texture.filter = (
                self.moderngl.NEAREST,
                self.moderngl.NEAREST,
            )
            depth_texture.repeat_x = False
            depth_texture.repeat_y = False
            depth_texture.compare_func = ""
            geometry_fbo = self.ctx.framebuffer(color_textures, depth_texture)
            clear_fbos = [
                self.ctx.framebuffer([color_textures[1]]),
                self.ctx.framebuffer([color_textures[2]]),
            ]
        except Exception:
            for framebuffer in reversed(clear_fbos):
                framebuffer.release()
            if geometry_fbo is not None:
                geometry_fbo.release()
            depth_texture.release()
            for texture in reversed(color_textures):
                texture.release()
            raise

        self.color_textures = color_textures
        self.depth_texture = depth_texture
        self.geometry_fbo = geometry_fbo
        self._attachment_clear_fbos = clear_fbos

    def _create_output_fbo(self, width: int, height: int) -> tuple[Any, Any]:
        texture = self.ctx.texture((int(width), int(height)), 4, dtype="f1")
        texture.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
        try:
            framebuffer = self.ctx.framebuffer([texture])
        except Exception:
            texture.release()
            raise
        return framebuffer, texture

    def _framebuffer_target(self, framebuffer: Any | int | None) -> Any:
        if framebuffer is None or (
            isinstance(framebuffer, int) and framebuffer == 0
        ):
            return self.ctx.screen
        return framebuffer

    def _render(self, mvp: np.ndarray, *, output_fbo: Any | int = 0) -> None:
        width, height = self.camera.width, self.camera.height
        mvp = np.ascontiguousarray(mvp, dtype=np.float32)

        background = tuple(float(value) for value in self.background)
        self.geometry_fbo.use()
        self.geometry_fbo.clear(*background, 1.0, depth=1.0)
        self._attachment_clear_fbos[0].use()
        self._attachment_clear_fbos[0].clear(*background, 0.0)
        self._attachment_clear_fbos[1].use()
        self._attachment_clear_fbos[1].clear(0.5, 0.5, 1.0, 0.0)
        self.geometry_fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.enable_only(self.moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"

        self.geom_program["uMVP"].write(_matrix_bytes(mvp))
        self.geom_program["uFaceForwardNormals"].value = (
            self.normal_orientation is NormalOrientation.FACE_FORWARD
        )
        for unit, texture in enumerate(self.lut_textures):
            texture.use(unit)
        self.geom_vao.render(
            mode=self.moderngl.TRIANGLES,
            vertices=self.vertex_count,
        )

        target = self._framebuffer_target(output_fbo)
        target.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(self.moderngl.DEPTH_TEST)
        self.post_program["uRenderMode"].value = int(self.render_mode)
        self.post_program["uNearClip"].value = self.camera.near_clip
        self.post_program["uFarClip"].value = self.camera.far_clip
        display_near, display_far = self._resolve_depth_range(mvp)
        self.post_program["uDepthDisplayNear"].value = display_near
        self.post_program["uDepthDisplayFar"].value = display_far
        for unit, texture in enumerate(self.color_textures):
            texture.use(unit)
        self.depth_texture.use(3)
        inverse = np.ascontiguousarray(np.linalg.inv(mvp), dtype=np.float32)
        self.post_program["uInvMVP"].write(_matrix_bytes(inverse))
        self.post_vao.render(mode=self.moderngl.TRIANGLES, vertices=3)

    def _resolve_depth_range(self, mvp: np.ndarray) -> tuple[float, float]:
        near_clip = float(self.camera.near_clip)
        far_clip = float(self.camera.far_clip)
        if self.render_mode is not RenderMode.DEPTH or self.depth_range is DepthRange.CLIP:
            result = (near_clip, far_clip)
        else:
            clips = (near_clip, far_clip)
            cache_hit = (
                self._depth_range_cache_mvp is not None
                and self._depth_range_cache_clips == clips
                and np.array_equal(self._depth_range_cache_mvp, mvp)
            )
            if cache_hit:
                result = (self.depth_display_near, self.depth_display_far)
            else:
                result = _automatic_depth_range(
                    self.scene.verts, mvp, near_clip, far_clip
                )
                self._depth_range_cache_mvp = np.array(mvp, dtype=np.float32, copy=True)
                self._depth_range_cache_clips = clips
        self.depth_display_near, self.depth_display_far = result
        return result

    def set_render_mode(self, mode: RenderMode | str | int) -> RenderMode:
        """Select color, linear depth, or world-space normal output."""
        self.render_mode = RenderMode.coerce(mode)
        return self.render_mode

    def set_depth_range(self, depth_range: DepthRange | str) -> DepthRange:
        """Select automatic contrast or fixed near/far clip normalization."""
        self.depth_range = DepthRange.coerce(depth_range)
        self._depth_range_cache_mvp = None
        self._depth_range_cache_clips = None
        return self.depth_range

    def set_normal_orientation(
        self, normal_orientation: NormalOrientation | str
    ) -> NormalOrientation:
        """Select view-facing two-sided normals or winding-oriented normals."""
        self.normal_orientation = NormalOrientation.coerce(normal_orientation)
        return self.normal_orientation

    def _draw_gui(self) -> None:
        imgui = self.imgui
        self.imgui_renderer.process_inputs()
        imgui.new_frame()
        imgui.set_next_window_position(20, 20, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(320, 360, condition=imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Settings", False)
        if expanded:
            imgui.text(f"Resolution: {self.camera.width}x{self.camera.height}")
            imgui.text(f"Faces: {len(self.scene.faces):,}")
            changed, mode = imgui.combo(
                "Render mode",
                int(self.render_mode),
                [candidate.label for candidate in RenderMode],
            )
            if changed:
                self.set_render_mode(mode)
            imgui.text("Shortcuts: 1 color, 2 depth, 3 normal")
            if self.render_mode is RenderMode.DEPTH:
                auto_depth = self.depth_range is DepthRange.AUTO
                changed, auto_depth = imgui.checkbox("Auto depth range", auto_depth)
                if changed:
                    self.set_depth_range(
                        DepthRange.AUTO if auto_depth else DepthRange.CLIP
                    )
                imgui.text(
                    "Linear camera Z: "
                    f"{self.depth_display_near:.3f} .. {self.depth_display_far:.3f}"
                )
                imgui.text("Near black, far white")
            elif self.render_mode is RenderMode.NORMAL:
                face_forward = (
                    self.normal_orientation is NormalOrientation.FACE_FORWARD
                )
                changed, face_forward = imgui.checkbox(
                    "Face-forward normals", face_forward
                )
                if changed:
                    self.set_normal_orientation(
                        NormalOrientation.FACE_FORWARD
                        if face_forward
                        else NormalOrientation.ORIENTED
                    )
                imgui.text("World-space RGB: X, Y, Z")
            changed, color = imgui.color_edit3("Background", *self.background)
            if changed:
                self.background[:] = color
            changed, value = imgui.slider_float("FOV", self.camera.fov_y_deg, 10.0, 120.0)
            if changed:
                self.camera.fov_y_deg = value
            changed, value = imgui.slider_float(
                "Near clip", self.camera.near_clip, 0.01, 50.0, format="%.2f"
            )
            if changed:
                self.camera.near_clip = min(value, self.camera.far_clip - 1e-3)
            changed, value = imgui.slider_float(
                "Far clip", self.camera.far_clip, 1.0, 1000.0, format="%.1f"
            )
            if changed:
                self.camera.far_clip = max(value, self.camera.near_clip + 1e-3)
            if imgui.button("Save screenshot"):
                self.save_screenshot()
        imgui.end()
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())

    def _mouse_is_captured(self) -> bool:
        return bool(self.imgui and self.imgui.get_io().want_capture_mouse)

    def _on_mouse_button(self, window, button: int, action: int, mods: int) -> None:
        if self.imgui_renderer:
            self.imgui_renderer.mouse_callback(window, button, action, mods)
        if action == self.glfw.RELEASE:
            self.camera.end_drag()
            return
        if self._mouse_is_captured():
            return
        glfw = self.glfw
        if action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(window)
            pan = (
                button in (glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE)
                or bool(mods & glfw.MOD_SHIFT)
            )
            self.camera.begin_drag(x, y, pan)
    def _on_cursor_pos(self, _window, x: float, y: float) -> None:
        if not self._mouse_is_captured():
            self.camera.drag_update(x, y)

    def _on_scroll(self, window, x_offset: float, y_offset: float) -> None:
        if self.imgui_renderer:
            self.imgui_renderer.scroll_callback(window, x_offset, y_offset)
        if not self._mouse_is_captured():
            self.camera.scroll(y_offset)

    def _on_resize(self, window, width: int, height: int) -> None:
        if self.imgui_renderer:
            self.imgui_renderer.resize_callback(window, width, height)
        if width > 0 and height > 0:
            self._resize_fbo(width, height)

    def _on_key(self, window, key: int, scancode: int, action: int, mods: int) -> None:
        if self.imgui_renderer:
            self.imgui_renderer.keyboard_callback(window, key, scancode, action, mods)
        if action != self.glfw.PRESS:
            return
        if key == self.glfw.KEY_ESCAPE:
            self.glfw.set_window_should_close(window, True)
        elif key == self.glfw.KEY_S:
            self.save_screenshot()
        elif key == self.glfw.KEY_1:
            self.set_render_mode(RenderMode.COLOR)
        elif key == self.glfw.KEY_2:
            self.set_render_mode(RenderMode.DEPTH)
        elif key == self.glfw.KEY_3:
            self.set_render_mode(RenderMode.NORMAL)

    def _on_char(self, window, codepoint: int) -> None:
        if self.imgui_renderer:
            self.imgui_renderer.char_callback(window, codepoint)

    def _read_pixels(self, framebuffer: Any | int = 0) -> np.ndarray:
        capture_fbo = None
        capture_texture = None
        is_screen = framebuffer is None or (
            isinstance(framebuffer, int) and framebuffer == 0
        )
        try:
            if is_screen:
                capture_fbo, capture_texture = self._create_output_fbo(
                    self.camera.width, self.camera.height
                )
                self.ctx.copy_framebuffer(capture_fbo, self.ctx.screen)
                target = capture_fbo
            else:
                target = framebuffer
            pixels = target.read(
                viewport=(0, 0, self.camera.width, self.camera.height),
                components=4,
                alignment=1,
            )
        finally:
            if capture_fbo is not None:
                capture_fbo.release()
            if capture_texture is not None:
                capture_texture.release()
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.camera.height, self.camera.width, 4
        )
        return np.flipud(image).copy()

    def save_screenshot(
        self,
        path: str | Path | None = None,
        *,
        framebuffer: Any | int = 0,
    ) -> Path:
        from PIL import Image

        target = Path(path) if path is not None else self.output_dir / "screenshot.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self._read_pixels(framebuffer), mode="RGBA").save(target)
        print(f"[py_viewer] saved {target}")
        return target

    def run(self) -> None:
        """Run the interactive event loop until the window is closed."""
        previous = time.perf_counter()
        frames = 0
        while not self.glfw.window_should_close(self.window):
            self.glfw.poll_events()
            self.camera.update()
            self._render(self.camera.mvp)
            self._draw_gui()
            self.glfw.swap_buffers(self.window)

            frames += 1
            now = time.perf_counter()
            if now - previous >= 0.5:
                fps = frames / (now - previous)
                self.glfw.set_window_title(
                    self.window,
                    f"DiffSoup Python Viewer  {self.render_mode.label}  "
                    f"{self.camera.width}x{self.camera.height}  FPS: {fps:.1f}",
                )
                frames = 0
                previous = now

    def run_benchmark(
        self,
        mvps: np.ndarray,
        *,
        warmup: int = 10,
        save_every: int = 0,
        repeat: int = 100,
    ) -> dict[str, float]:
        """Benchmark column-major MVP payloads using the native timing protocol."""
        mvps = np.ascontiguousarray(mvps, dtype=np.float32)
        if mvps.ndim != 3 or mvps.shape[1:] != (4, 4) or len(mvps) == 0:
            raise ValueError("mvps must be a non-empty float32 [B, 4, 4] array")
        if repeat < 1:
            raise ValueError("repeat must be positive")

        output_fbo, output_texture = self._create_output_fbo(
            self.camera.width, self.camera.height
        )
        screenshots = self.output_dir / "screenshots"
        screenshots.mkdir(parents=True, exist_ok=True)
        try:
            first_mvp = np.ascontiguousarray(mvps[0].T)
            for _ in range(max(0, warmup)):
                self._render(first_mvp, output_fbo=output_fbo)
            self.ctx.finish()

            times_ms: list[float] = []
            for index, payload in enumerate(mvps):
                mvp = np.ascontiguousarray(payload.T)
                self.ctx.finish()
                started = time.perf_counter()
                for _ in range(repeat):
                    self._render(mvp, output_fbo=output_fbo)
                self.ctx.finish()
                elapsed_ms = (time.perf_counter() - started) * 1000.0 / repeat
                times_ms.append(elapsed_ms)
                self.glfw.poll_events()

                if save_every > 0 and index % save_every == 0:
                    self.save_screenshot(
                        screenshots / f"benchmark_{index:05d}.png",
                        framebuffer=output_fbo,
                    )

            values = np.asarray(times_ms, dtype=np.float64)
            summary = {
                "frames": float(len(values)),
                "mean_ms": float(values.mean()),
                "min_ms": float(values.min()),
                "max_ms": float(values.max()),
                "fps": float(1000.0 / values.mean()),
            }
            with (self.output_dir / "benchmark_frames.txt").open("w", encoding="utf-8") as file:
                for index, value in enumerate(values):
                    file.write(f"{index} {value}\n")
            with (self.output_dir / "benchmark_summary.txt").open("w", encoding="utf-8") as file:
                file.write(
                    f"frames: {len(values)}\n"
                    f"mean_ms: {summary['mean_ms']}\n"
                    f"min_ms:  {summary['min_ms']}\n"
                    f"max_ms:  {summary['max_ms']}\n"
                    f"fps:     {summary['fps']}\n"
                )
            return summary
        finally:
            output_fbo.release()
            output_texture.release()

    def _release_gl_resources(self) -> None:
        if self.ctx is None:
            return
        self._release_geometry_targets()
        for texture in reversed(self.lut_textures):
            texture.release()
        self.lut_textures = []
        for name in ("post_vao", "geom_vao"):
            resource = getattr(self, name)
            if resource is not None:
                resource.release()
                setattr(self, name, None)
        for name in (
            "normal_buffer",
            "triangle_id_buffer",
            "position_buffer",
        ):
            resource = getattr(self, name)
            if resource is not None:
                resource.release()
                setattr(self, name, None)
        for name in ("post_program", "geom_program"):
            resource = getattr(self, name)
            if resource is not None:
                resource.release()
                setattr(self, name, None)
        self.ctx.release()
        self.ctx = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.window:
            self.glfw.make_context_current(self.window)
            if self.imgui_renderer is not None:
                self.imgui_renderer.shutdown()
                self.imgui_renderer = None
            if self.imgui_context is not None:
                self.imgui.destroy_context(self.imgui_context)
                self.imgui_context = None
            self._release_gl_resources()
            self.glfw.destroy_window(self.window)
            self.window = None
        self.glfw.terminate()

    def __enter__(self) -> "Viewer":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def launch_scene(
    scene: SceneData,
    *,
    output_dir: str | Path = "./results/py_viewer",
    width: int = 1200,
    height: int = 1200,
    render_mode: RenderMode | str | int = RenderMode.COLOR,
    depth_range: DepthRange | str = DepthRange.AUTO,
    normal_orientation: NormalOrientation | str = NormalOrientation.FACE_FORWARD,
) -> None:
    """Launch an interactive viewer for a validated scene."""
    with Viewer(
        scene, width=width, height=height, output_dir=output_dir,
        interactive=True, render_mode=render_mode, depth_range=depth_range,
        normal_orientation=normal_orientation,
    ) as app:
        app.run()


def launch_viewer(
    verts: np.ndarray,
    faces: np.ndarray,
    face_color_lut: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    output_dir: str | Path = "./results/py_viewer",
    up: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    level: int = 5,
    width: int = 1200,
    height: int = 1200,
    render_mode: RenderMode | str | int = RenderMode.COLOR,
    depth_range: DepthRange | str = DepthRange.AUTO,
    normal_orientation: NormalOrientation | str = NormalOrientation.FACE_FORWARD,
) -> None:
    """Array-based entry point corresponding to ``diffsoupviewer.launch_viewer``."""
    scene = SceneData.from_face_color_lut(
        verts, faces, face_color_lut, w1, b1, w2, b2, w3, b3,
        level=level, up=up,
    )
    launch_scene(
        scene, output_dir=output_dir, width=width, height=height,
        render_mode=render_mode, depth_range=depth_range,
        normal_orientation=normal_orientation,
    )


def benchmark(
    verts: np.ndarray,
    faces: np.ndarray,
    lut0: np.ndarray,
    lut1: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    mvps: np.ndarray,
    width: int = 1200,
    height: int = 1200,
    warmup: int = 10,
    save_every: int = 0,
    output_dir: str | Path = "./results/py_viewer",
    up: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    level: int = 5,
    repeat: int = 100,
    render_mode: RenderMode | str | int = RenderMode.COLOR,
    depth_range: DepthRange | str = DepthRange.AUTO,
    normal_orientation: NormalOrientation | str = NormalOrientation.FACE_FORWARD,
) -> dict[str, float]:
    """Array-based benchmark corresponding to ``diffsoupviewer.benchmark``."""
    scene = SceneData(
        verts, faces, lut0, lut1, w1, b1, w2, b2, w3, b3,
        level=level, up=up,
    )
    with Viewer(
        scene, width=width, height=height, output_dir=output_dir,
        interactive=False, render_mode=render_mode, depth_range=depth_range,
        normal_orientation=normal_orientation,
    ) as app:
        return app.run_benchmark(
            mvps, warmup=warmup, save_every=save_every, repeat=repeat
        )
