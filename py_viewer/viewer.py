"""GLFW/PyOpenGL implementation of the DiffSoup native viewer."""

from __future__ import annotations

import ctypes
import importlib
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .camera import OrbitCamera
from .scene import SceneData


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
        gl = importlib.import_module("OpenGL.GL")
        gl_shaders = importlib.import_module("OpenGL.GL.shaders")
    except ImportError as exc:
        raise RuntimeError(
            "The Python viewer requires glfw and PyOpenGL. "
            "Install them with `pip install glfw PyOpenGL`."
        ) from exc

    imgui = None
    glfw_renderer = None
    if interactive:
        try:
            imgui = importlib.import_module("imgui")
            integration = importlib.import_module("imgui.integrations.glfw")
        except ImportError as exc:
            raise RuntimeError(
                "Interactive mode requires pyimgui. "
                "Install it with `pip install imgui`."
            ) from exc
        glfw_renderer = integration.GlfwRenderer
    return glfw, gl, gl_shaders, imgui, glfw_renderer


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
    ) -> None:
        self.scene = scene
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        self.background = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self.glfw, self.gl, self.gl_shaders, self.imgui, renderer_cls = (
            _load_runtime(interactive)
        )
        self.window = None
        self.imgui_renderer = None
        self._closed = False

        self.camera = OrbitCamera(
            width=max(1, int(width)),
            height=max(1, int(height)),
            world_up=np.asarray(scene.up, dtype=np.float32),
            target=scene.center,
        )

        self._create_window(width, height, visible=interactive)
        if interactive:
            self.imgui.create_context()
            self.imgui_renderer = renderer_cls(self.window, attach_callbacks=False)

        self._install_callbacks()
        self._init_gl()

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
        self.window = glfw.create_window(int(width), int(height), "DiffSoup Python Viewer", None, None)
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

    def _compile_program(self, vertex: str, fragment: str) -> int:
        gl = self.gl
        return int(
            self.gl_shaders.compileProgram(
                self.gl_shaders.compileShader(vertex, gl.GL_VERTEX_SHADER),
                self.gl_shaders.compileShader(fragment, gl.GL_FRAGMENT_SHADER),
            )
        )

    def _init_gl(self) -> None:
        gl = self.gl
        gl.glClearDepth(1.0)
        gl.glDepthFunc(gl.GL_LESS)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        self.geom_program = self._compile_program(
            _load_shader_source("geometry.vert.glsl"),
            _load_shader_source("geometry.frag.glsl"),
        )
        self.post_program = self._compile_program(
            _load_shader_source("post.vert.glsl"),
            _load_shader_source("post.frag.glsl"),
        )

        self.geom_uniforms = {
            name: gl.glGetUniformLocation(self.geom_program, name)
            for name in ("uMVP", "uTriTexSize", "uTriTex0", "uTriTex1", "uLevel")
        }
        self.post_uniforms = {
            name: gl.glGetUniformLocation(self.post_program, name)
            for name in (
                "texA", "texB", "uInvMVP", "W1[0]", "B1[0]",
                "W2[0]", "B2[0]", "W3[0]", "B3",
            )
        }

        self.geom_vao = int(gl.glGenVertexArrays(1))
        self.position_vbo = int(gl.glGenBuffers(1))
        self.triangle_id_vbo = int(gl.glGenBuffers(1))
        self.post_vao = int(gl.glGenVertexArrays(1))
        self._upload_mesh()
        self.lut_textures = [self._upload_texture(self.scene.lut0), self._upload_texture(self.scene.lut1)]
        self._upload_mlp()

        self.fbo = int(gl.glGenFramebuffers(1))
        self.color_textures = [int(gl.glGenTextures(1)), int(gl.glGenTextures(1))]
        self.depth_texture = int(gl.glGenTextures(1))
        width, height = self.glfw.get_framebuffer_size(self.window)
        self._resize_fbo(max(1, width), max(1, height))

    def _upload_mesh(self) -> None:
        gl = self.gl
        positions = np.ascontiguousarray(
            self.scene.verts[self.scene.faces].reshape(-1, 3), dtype=np.float32
        )
        triangle_ids = np.ascontiguousarray(
            np.repeat(np.arange(len(self.scene.faces), dtype=np.uint32), 3)
        )
        self.vertex_count = int(len(positions))

        gl.glBindVertexArray(self.geom_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.position_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, positions.nbytes, positions, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.triangle_id_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, triangle_ids.nbytes, triangle_ids, gl.GL_STATIC_DRAW
        )
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribIPointer(1, 1, gl.GL_UNSIGNED_INT, 0, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

    def _upload_texture(self, rgba: np.ndarray) -> int:
        gl = self.gl
        texture = int(gl.glGenTextures(1))
        height, width, _ = rgba.shape
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, rgba,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return texture

    def _upload_mlp(self) -> None:
        gl = self.gl
        gl.glUseProgram(self.post_program)
        gl.glUniformMatrix4fv(
            self.post_uniforms["W1[0]"], 16, gl.GL_TRUE, _tile_weights(self.scene.w1)
        )
        gl.glUniform4fv(
            self.post_uniforms["B1[0]"], 4, np.ascontiguousarray(self.scene.b1.reshape(4, 4))
        )
        gl.glUniformMatrix4fv(
            self.post_uniforms["W2[0]"], 16, gl.GL_TRUE, _tile_weights(self.scene.w2)
        )
        gl.glUniform4fv(
            self.post_uniforms["B2[0]"], 4, np.ascontiguousarray(self.scene.b2.reshape(4, 4))
        )
        gl.glUniformMatrix4fv(
            self.post_uniforms["W3[0]"], 4, gl.GL_TRUE, _tile_output_weights(self.scene.w3)
        )
        b3 = np.array([*self.scene.b3, 0.0], dtype=np.float32)
        gl.glUniform4fv(self.post_uniforms["B3"], 1, b3)
        gl.glUseProgram(0)

    def _resize_fbo(self, width: int, height: int) -> None:
        gl = self.gl
        width, height = max(1, int(width)), max(1, int(height))
        self.camera.width = width
        self.camera.height = height
        self.camera.update()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        for index, texture in enumerate(self.color_textures):
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0,
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
            )
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0 + index,
                gl.GL_TEXTURE_2D, texture, 0,
            )

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, width, height, 0,
            gl.GL_DEPTH_COMPONENT, gl.GL_UNSIGNED_INT, None,
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
            gl.GL_TEXTURE_2D, self.depth_texture, 0,
        )
        gl.glDrawBuffers(2, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1])
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("geometry framebuffer is incomplete")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _create_output_fbo(self, width: int, height: int) -> tuple[int, int]:
        gl = self.gl
        fbo = int(gl.glGenFramebuffers(1))
        texture = int(gl.glGenTextures(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None,
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D, texture, 0,
        )
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("benchmark framebuffer is incomplete")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return fbo, texture

    def _render(self, mvp: np.ndarray, *, output_fbo: int = 0) -> None:
        gl = self.gl
        width, height = self.camera.width, self.camera.height
        mvp = np.ascontiguousarray(mvp, dtype=np.float32)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, width, height)
        gl.glClearBufferfv(
            gl.GL_COLOR, 0, np.array([*self.background, 1.0], dtype=np.float32)
        )
        gl.glClearBufferfv(
            gl.GL_COLOR, 1, np.array([*self.background, 0.0], dtype=np.float32)
        )
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)

        gl.glUseProgram(self.geom_program)
        gl.glUniformMatrix4fv(self.geom_uniforms["uMVP"], 1, gl.GL_TRUE, mvp)
        gl.glUniform1i(self.geom_uniforms["uTriTex0"], 0)
        gl.glUniform1i(self.geom_uniforms["uTriTex1"], 1)
        gl.glUniform2i(
            self.geom_uniforms["uTriTexSize"],
            int(self.scene.lut0.shape[1]), int(self.scene.lut0.shape[0]),
        )
        gl.glUniform1i(self.geom_uniforms["uLevel"], self.scene.level)
        for unit, texture in enumerate(self.lut_textures):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glBindVertexArray(self.geom_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertex_count)
        gl.glBindVertexArray(0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, output_fbo)
        gl.glViewport(0, 0, width, height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(self.post_program)
        gl.glUniform1i(self.post_uniforms["texA"], 0)
        gl.glUniform1i(self.post_uniforms["texB"], 1)
        for unit, texture in enumerate(self.color_textures):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        inverse = np.ascontiguousarray(np.linalg.inv(mvp), dtype=np.float32)
        gl.glUniformMatrix4fv(self.post_uniforms["uInvMVP"], 1, gl.GL_TRUE, inverse)
        gl.glBindVertexArray(self.post_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def _draw_gui(self) -> None:
        imgui = self.imgui
        self.imgui_renderer.process_inputs()
        imgui.new_frame()
        imgui.set_next_window_position(20, 20, condition=imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(300, 260, condition=imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Settings", False)
        if expanded:
            imgui.text(f"Resolution: {self.camera.width}x{self.camera.height}")
            imgui.text(f"Faces: {len(self.scene.faces):,}")
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

    def _on_char(self, window, codepoint: int) -> None:
        if self.imgui_renderer:
            self.imgui_renderer.char_callback(window, codepoint)

    def _read_pixels(self, framebuffer: int) -> np.ndarray:
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, framebuffer)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 if framebuffer else gl.GL_BACK)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        pixels = gl.glReadPixels(
            0, 0, self.camera.width, self.camera.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE
        )
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(
            self.camera.height, self.camera.width, 4
        )
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        return np.flipud(image).copy()

    def save_screenshot(
        self,
        path: str | Path | None = None,
        *,
        framebuffer: int = 0,
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
                    f"DiffSoup Python Viewer  {self.camera.width}x{self.camera.height}  FPS: {fps:.1f}",
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
            self.gl.glFinish()

            times_ms: list[float] = []
            for index, payload in enumerate(mvps):
                mvp = np.ascontiguousarray(payload.T)
                self.gl.glFinish()
                started = time.perf_counter()
                for _ in range(repeat):
                    self._render(mvp, output_fbo=output_fbo)
                self.gl.glFinish()
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
            self.gl.glDeleteTextures(1, [output_texture])
            self.gl.glDeleteFramebuffers(1, [output_fbo])

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.window:
            self.glfw.make_context_current(self.window)
            if self.imgui_renderer is not None:
                self.imgui_renderer.shutdown()
                self.imgui.destroy_context()
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
) -> None:
    """Launch an interactive viewer for a validated scene."""
    with Viewer(
        scene, width=width, height=height, output_dir=output_dir, interactive=True
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
) -> None:
    """Array-based entry point corresponding to ``diffsoupviewer.launch_viewer``."""
    scene = SceneData.from_face_color_lut(
        verts, faces, face_color_lut, w1, b1, w2, b2, w3, b3,
        level=level, up=up,
    )
    launch_scene(scene, output_dir=output_dir, width=width, height=height)


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
) -> dict[str, float]:
    """Array-based benchmark corresponding to ``diffsoupviewer.benchmark``."""
    scene = SceneData(
        verts, faces, lut0, lut1, w1, b1, w2, b2, w3, b3,
        level=level, up=up,
    )
    with Viewer(
        scene, width=width, height=height, output_dir=output_dir, interactive=False
    ) as app:
        return app.run_benchmark(
            mvps, warmup=warmup, save_every=save_every, repeat=repeat
        )
