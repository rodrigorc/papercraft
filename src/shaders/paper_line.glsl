#version 140

uniform mat3 m;
in vec2 pos;
in float line_dash;

out float v_line_dash;

void main(void) {
    gl_Position = vec4((m * vec3(pos, 1.0)).xy, 0.0, 1.0);
    v_line_dash = line_dash;
}

###

#version 140

uniform vec4 line_color;
uniform float frac_dash;

in float v_line_dash;
out vec4 out_frag_color;

void main(void) {
    float alpha = 1.0 - step(frac_dash, mod(v_line_dash, 1.0));
    out_frag_color = vec4(line_color.rgb, line_color.a * alpha);
}
