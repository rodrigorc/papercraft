#version 140

uniform mat4 m;
uniform int line_top;

in vec3 pos;
in vec4 color;

out vec4 v_color;

void main(void) {
    v_color = color;
    gl_Position = m * vec4(pos, 1.0);
    if (line_top != 0) {
        gl_Position.z = 0.0;
    }
}

###

#version 140

in vec4 v_color;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = v_color;
}
