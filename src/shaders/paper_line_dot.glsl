#version 150

uniform mat3 m;

in vec2 pos;
in vec2 uv;

out float v_line_pos;

void main(void) {
    gl_Position = vec4((m * vec3(pos, 1.0)).xy, 0.0, 1.0);
    v_line_pos = uv.x;
}

###

#version 150

uniform float frac_dash;
in float v_line_pos;
out vec4 out_frag_color;

void main(void) {
    float alpha = step(frac_dash, mod(v_line_pos, 1.0));
    out_frag_color = vec4(0.0, 0.0, 0.0, alpha);
}
