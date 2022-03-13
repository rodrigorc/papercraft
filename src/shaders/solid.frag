#version 150

uniform sampler2D tex;

in vec2 v_uv;
in float v_light;
out vec4 out_frag_color;

void main(void) {
    vec4 base;
    if (gl_FrontFacing)
        base = texture2D(tex, v_uv);
    else
        base = vec4(0.8, 0.3, 0.3, 1.0);
    out_frag_color = vec4(v_light * base.rgb, base.a);
}
