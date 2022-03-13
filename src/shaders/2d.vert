#version 150

uniform mat3 m;

in vec2 pos;
in vec2 uv;

out vec2 v_uv;
out float v_light;

void main(void) {
    gl_Position = vec4((m * vec3(pos, 1.0)).xy, 0.0, 1.0);
    v_light = 1.0;
    v_uv = uv;
}
