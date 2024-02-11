#version 150
precision highp float;

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
precision highp float;

uniform sampler2D tex;

in vec2 v_uv;
out vec4 out_frag_color;

void main(void) {
    vec4 c = texture(tex, v_uv);
    out_frag_color = vec4(0.0, 0.0, 0.0, c.a);
}
