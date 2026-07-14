#version 410 core
flat in uint vTriID;
flat in vec3 vNormal;
in vec3 vBary;

layout(location=0) out vec4 FragA;
layout(location=1) out vec4 FragB;
layout(location=2) out vec4 FragNormal;

uniform ivec2 uTriTexSize;
uniform sampler2D uTriTex0;
uniform sampler2D uTriTex1;
uniform int uLevel;
uniform bool uFaceForwardNormals;

ivec2 idx_to_coord(int idx, int texW) {
    return ivec2(idx % texW, idx / texW);
}

int level_size(int level) {
    if (level == 0) return 3;
    int a = (1 << (level - 1)) + 1;
    int b = (1 << level) + 1;
    return a * b;
}

void main() {
    int texW = uTriTexSize.x;
    int texH = uTriTexSize.y;
    if (texW * texH <= 0) {
        FragA = FragB = vec4(1, 0, 1, 1);
        FragNormal = vec4(0.5, 0.5, 1.0, 1.0);
        return;
    }

    int samples = level_size(uLevel);
    int base = int(vTriID) * samples;

    float b0 = vBary.x;
    float b1 = vBary.y;
    int res = 1 << uLevel;
    float res_f = float(res);

    float b0l = b0 * res_f;
    float b1l = b1 * res_f;
    int x = clamp(int(floor(b0l)), 0, res - 1);
    int y = clamp(int(floor(b1l)), 0, (res - 1) - x);
    b0l -= float(x);
    b1l -= float(y);

    bool flip = (b0l + b1l) > 1.0;
    int flip_u = flip ? 1 : 0;
    float flip_f = flip ? 1.0 : 0.0;

    int x0 = x + 1,      y0 = y;
    int x1 = x,          y1 = y + 1;
    int x2 = x + flip_u, y2 = min(y + flip_u, res - x2);

    int idx0 = (x0 + y0) * (x0 + y0 + 1) / 2 + y0;
    int idx1 = (x1 + y1) * (x1 + y1 + 1) / 2 + y1;
    int idx2 = (x2 + y2) * (x2 + y2 + 1) / 2 + y2;

    float w0 = mix(b0l, 1.0 - b1l, flip_f);
    float w1 = mix(b1l, 1.0 - b0l, flip_f);
    float w2 = 1.0 - w0 - w1;

    ivec2 c0 = idx_to_coord(base + idx0, texW);
    ivec2 c1 = idx_to_coord(base + idx1, texW);
    ivec2 c2 = idx_to_coord(base + idx2, texW);

    vec4 a = texelFetch(uTriTex0, c0, 0) * w0
           + texelFetch(uTriTex0, c1, 0) * w1
           + texelFetch(uTriTex0, c2, 0) * w2;
    vec4 b = texelFetch(uTriTex1, c0, 0) * w0
           + texelFetch(uTriTex1, c1, 0) * w1
           + texelFetch(uTriTex1, c2, 0) * w2;

    if (b.a < 0.5) discard;
    FragA = a;
    FragB = vec4(b.rgb, 1.0);
    vec3 normal = normalize(vNormal);
    if (uFaceForwardNormals && !gl_FrontFacing) normal = -normal;
    FragNormal = vec4(normal * 0.5 + 0.5, 1.0);
}
