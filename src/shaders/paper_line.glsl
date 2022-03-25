#version 150

uniform mat3 m;

in vec2 pos;
in vec2 uv;
in vec4 color;

out vec4 v_color;
out float v_line_pos;

void main(void) {
    gl_Position = vec4((m * vec3(pos, 1.0)).xy, 0.0, 1.0);
    v_line_pos = uv.x;
    v_color = color;
}

###

#version 150

uniform float frac_dash;
in vec4 v_color;
in float v_line_pos;
out vec4 out_frag_color;

void main(void) {
    float alpha = 1.0 - step(frac_dash, mod(v_line_pos, 1.0));
    out_frag_color = vec4(v_color.rgb, v_color.a * alpha);
}
