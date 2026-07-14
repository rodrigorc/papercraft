#version 140

uniform mat3 m;

in vec2 pos_2d;
in vec4 color;
in float line_dash;
in int valley;

out float v_line_dash;
out vec4 v_color;
flat out int v_valley;

void main(void) {
    v_line_dash = line_dash;
    v_color = color;
    v_valley = valley;
    gl_Position = vec4((m * vec3(pos_2d, 1.0)).xy, 0.0, 1.0);
}

###

#version 140

uniform sampler1D tex;
uniform sampler1D tex_2;

in vec4 v_color;
in float v_line_dash;
flat in int v_valley;
out vec4 out_frag_color;

void main(void) {
    float alpha;
    if (v_valley != 0)
        alpha = texture(tex, v_line_dash).r;
    else
        alpha = texture(tex_2, v_line_dash).r;
    out_frag_color = vec4(v_color.rgb, v_color.a * alpha);
}
