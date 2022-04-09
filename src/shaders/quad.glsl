#version 140

in vec2 pos;

void main(void) {
    gl_Position = vec4(pos, 0.0, 1.0);
}

###

#version 140

uniform vec4 color;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = color;
}
