#version 150

uniform mat3 m;

in vec2 pos;
in vec2 uv;

out vec2 v_uv;

void main(void) {
    gl_Position = vec4((m * vec3(pos, 1.0)).xy, 0.0, 1.0);
    v_uv = uv;
}

###

#version 150

uniform sampler2D tex;
uniform vec4 color;

in vec2 v_uv;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = mix(texture2D(tex, v_uv), color, color.a);
}
