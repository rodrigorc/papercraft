#version 140

uniform mat3 m;

in vec2 pos_2d;
in vec4 color;
in float line_dash;

out float v_line_dash;
out vec4 v_color;

void main(void) {
    v_line_dash = line_dash;
    v_color = color;
    gl_Position = vec4((m * vec3(pos_2d, 1.0)).xy, 0.0, 1.0);
}

###

#version 140

in vec4 v_color;
in float v_line_dash;
out vec4 out_frag_color;

void main(void) {
    float alpha = 1.0 - step(0.5, mod(v_line_dash, 1.0));
    out_frag_color = vec4(v_color.rgb, v_color.a * alpha);
}
