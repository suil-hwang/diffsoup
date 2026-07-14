#version 410 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D texA;
uniform sampler2D texB;
uniform mat4 uInvMVP;

uniform mat4 W1[16];
uniform vec4 B1[4];
uniform mat4 W2[16];
uniform vec4 B2[4];
uniform mat4 W3[4];
uniform vec4 B3;

vec4 relu4(vec4 x) { return max(x, 0.0); }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

vec3 ndc_to_world(vec4 ndc) {
    vec4 clip = uInvMVP * ndc;
    float w = clip.w;
    if (abs(w) < 1e-20) w = 1e-20;
    return clip.xyz / w;
}

const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2_0 = 1.0925484305920792;
const float SH_C2_1 = -1.0925484305920792;
const float SH_C2_2 = 0.31539156525252005;
const float SH_C2_3 = -1.0925484305920792;
const float SH_C2_4 = 0.5462742152960396;

void eval_sh2(vec3 d, out float sh[9]) {
    sh[0] = SH_C0;
    sh[1] = -SH_C1 * d.y;
    sh[2] = SH_C1 * d.z;
    sh[3] = -SH_C1 * d.x;
    float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
    sh[4] = SH_C2_0 * d.x * d.y;
    sh[5] = SH_C2_1 * d.y * d.z;
    sh[6] = SH_C2_2 * (2.0 * zz - xx - yy);
    sh[7] = SH_C2_3 * d.x * d.z;
    sh[8] = SH_C2_4 * (xx - yy);
}

void main() {
    vec4 A = texture(texA, vUV);
    vec4 B = texture(texB, vUV);
    if (B.a < 0.5) {
        FragColor = vec4(A.rgb, 1.0);
        return;
    }

    float ndc_x = vUV.x * 2.0 - 1.0;
    float ndc_y = vUV.y * 2.0 - 1.0;
    vec3 world_near = ndc_to_world(vec4(ndc_x, ndc_y, -1.0, 1.0));
    vec3 world_far = ndc_to_world(vec4(ndc_x, ndc_y, 1.0, 1.0));
    vec3 view_dir = normalize(world_near - world_far);

    float sh[9];
    eval_sh2(view_dir, sh);

    vec4 x0 = vec4(A.r, A.g, A.b, A.a);
    vec4 x1 = vec4(B.r, B.g, B.b, sh[0]);
    vec4 x2 = vec4(sh[1], sh[2], sh[3], sh[4]);
    vec4 x3 = vec4(sh[5], sh[6], sh[7], sh[8]);

    vec4 y0 = relu4(W1[0]*x0 + W1[1]*x1 + W1[2]*x2 + W1[3]*x3 + B1[0]);
    vec4 y1 = relu4(W1[4]*x0 + W1[5]*x1 + W1[6]*x2 + W1[7]*x3 + B1[1]);
    vec4 y2 = relu4(W1[8]*x0 + W1[9]*x1 + W1[10]*x2 + W1[11]*x3 + B1[2]);
    vec4 y3 = relu4(W1[12]*x0 + W1[13]*x1 + W1[14]*x2 + W1[15]*x3 + B1[3]);

    vec4 z0 = relu4(W2[0]*y0 + W2[1]*y1 + W2[2]*y2 + W2[3]*y3 + B2[0]);
    vec4 z1 = relu4(W2[4]*y0 + W2[5]*y1 + W2[6]*y2 + W2[7]*y3 + B2[1]);
    vec4 z2 = relu4(W2[8]*y0 + W2[9]*y1 + W2[10]*y2 + W2[11]*y3 + B2[2]);
    vec4 z3 = relu4(W2[12]*y0 + W2[13]*y1 + W2[14]*y2 + W2[15]*y3 + B2[3]);

    vec4 logits = W3[0]*z0 + W3[1]*z1 + W3[2]*z2 + W3[3]*z3 + B3;
    vec3 mlp = vec3(sigmoid(logits.x), sigmoid(logits.y), sigmoid(logits.z));
    FragColor = vec4(mix(A.rgb, mlp, A.a), 1.0);
}
