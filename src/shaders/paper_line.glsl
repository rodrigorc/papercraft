#version 150

in vec2 pos;
in float line_dash;
in float width_left;
in float width_right;

out Vertex {
    float line_dash;
    float width_left;
    float width_right;
    vec2 pos;
} v_data;

void main(void) {
    v_data.pos = pos;
    v_data.line_dash = line_dash;
    v_data.width_left = width_left;
    v_data.width_right = width_right;
}

###

#version 150

uniform vec4 line_color;
uniform float frac_dash;

in float v_line_dash;
out vec4 out_frag_color;

void main(void) {
    float alpha = 1.0 - step(frac_dash, mod(v_line_dash, 1.0));
    out_frag_color = vec4(line_color.rgb, line_color.a * alpha);
}

###

#version 150

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat3 m;
in Vertex {
    float line_dash;
    float width_left;
    float width_right;
    vec2 pos;
} v_data[];
out float v_line_dash;

vec4 mx(vec2 p) {
    vec3 t = m * vec3(p, 1.0);
    return vec4(t.xy, 0.0, 1.0);
}

void main(void) {
    vec2 p0 = v_data[0].pos.xy;
    vec2 p1 = v_data[1].pos.xy;

    vec2 v = p1 - p0;
    v /= length(v);
    vec2 n = vec2(v.y, -v.x);
    v *= (v_data[0].width_right + v_data[0].width_left) / 2.0;
    p0 -= v;
    p1 += v;

    gl_Position = mx(p0 + n * v_data[0].width_right);
    v_line_dash = v_data[0].line_dash;
    EmitVertex();

    gl_Position = mx(p1 + n * v_data[1].width_right);
    v_line_dash = v_data[1].line_dash;
    EmitVertex();

    gl_Position = mx(p0 - n * v_data[0].width_left);
    v_line_dash = v_data[0].line_dash;
    EmitVertex();

    gl_Position = mx(p1 - n * v_data[1].width_left);
    v_line_dash = v_data[1].line_dash;
    EmitVertex();

    EndPrimitive();
}
