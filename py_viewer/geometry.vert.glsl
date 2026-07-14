#version 410 core
layout(location=0) in vec3 aPos;
layout(location=1) in uint aTriID;

flat out uint vTriID;
out vec3 vBary;

uniform mat4 uMVP;

void main() {
    vTriID = aTriID;
    int corner = gl_VertexID % 3;
    vBary = (corner == 0) ? vec3(1, 0, 0) :
            (corner == 1) ? vec3(0, 1, 0) :
                            vec3(0, 0, 1);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
