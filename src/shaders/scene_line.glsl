#version 150

uniform mat4 m;
in vec3 pos;

void main(void) {
    gl_Position = m * vec4(pos, 1.0);
}

###

#version 150

uniform vec4 color;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = color;
}
