#version 140

uniform mat3 m;

in vec2 pos_2d;
in vec2 uv;
in float mat;
in vec4 color;

out vec3 v_uv;
out vec4 v_color;

void main(void) {
    gl_Position = vec4((m * vec3(pos_2d, 1.0)).xy, 0.0, 1.0);
    v_uv = vec3(uv, mat);
    v_color = color;
}

###

#version 140

uniform bool texturize;
uniform vec4 notex_color;
uniform sampler2DArray tex;

in vec3 v_uv;
in vec4 v_color;
out vec4 out_frag_color;

void main(void) {
    vec4 c;
    if (texturize) {
        c = texture(tex, v_uv);
    } else {
        c = notex_color;
    }
    out_frag_color = mix(c, vec4(v_color.rgb, 1.0), v_color.a);
}
