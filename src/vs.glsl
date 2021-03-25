#version 330 core

uniform mat4 transform;

layout(location=0) in vec2 vs_pos;
layout(location=1) in vec2 vs_uv;
layout(location=2) in vec4 vs_color;

out vec4 fs_color;
out vec2 fs_uv;

void main() {
    gl_Position = transform * vec4(vs_pos, 0.0, 1.0);
    fs_color = vs_color;
    fs_uv = vs_uv;
}
