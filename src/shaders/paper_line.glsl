#version 140

in vec2 pos;
in float line_dash;
in float width_left;
in float width_right;

out float v_line_dash;
out float v_width_left;
out float v_width_right;

void main(void) {
    gl_Position.xy = pos;
    v_line_dash = line_dash;
    v_width_left = width_left;
    v_width_right = width_right;
}

###

#version 140

uniform vec4 line_color;
uniform float frac_dash;

in float v_line_dash_;
out vec4 out_frag_color;

void main(void) {
    float alpha = 1.0 - step(frac_dash, mod(v_line_dash_, 1.0));
    out_frag_color = vec4(line_color.rgb, line_color.a * alpha);
}

###

#version 150

layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

in gl_PerVertex
{
  vec2 gl_Position;
} gl_in[];

out gl_PerVertex
{
  vec3 gl_Position;
};

uniform mat3 m;
in float v_width_left[];
in float v_width_right[];
in float v_line_dash[];
out float v_line_dash_;


vec3 mx(vec2 p)
{
    return m * vec3(p, 1.0);
}

void main(void) {
    vec2 p0 = gl_in[0].gl_Position;
    vec2 p1 = gl_in[1].gl_Position;

    vec2 v = p1 - p0;
    v /= length(v);
    vec2 n = vec2(v.y, -v.x);
    v *= (v_width_right[0] + v_width_left[0]) / 2.0;
    p0 -= v;
    p1 += v;

    gl_Position = mx(p0 + n * v_width_right[0]);
    v_line_dash_ = v_line_dash[0];
    EmitVertex();

    gl_Position = mx(p1 + n * v_width_right[1]);
    v_line_dash_ = v_line_dash[1];
    EmitVertex();

    gl_Position = mx(p0 - n * v_width_left[0]);
    v_line_dash_ = v_line_dash[0];
    EmitVertex();

    gl_Position = mx(p1 - n * v_width_left[1]);
    v_line_dash_ = v_line_dash[1];
    EmitVertex();

    EndPrimitive();
}