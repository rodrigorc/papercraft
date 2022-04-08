#version 150

uniform mat3 m;

in vec2 pos;
in vec2 uv;
in vec4 color;

out vec2 v_uv;
out vec4 v_color;

void main(void) {
    gl_Position = vec4((m * vec3(pos, 1.0)).xy, 0.0, 1.0);
    v_uv = uv;
    v_color = color;
}

###

#version 150

uniform sampler2D tex;

in vec2 v_uv;
in vec4 v_color;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = mix(texture2D(tex, v_uv), vec4(v_color.rgb, 1.0), v_color.a);
}
