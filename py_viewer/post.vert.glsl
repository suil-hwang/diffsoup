#version 410 core
const vec2 verts[3] = vec2[3](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));
out vec2 vUV;

void main() {
    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
    vUV = 0.5 * (gl_Position.xy + 1.0);
}
