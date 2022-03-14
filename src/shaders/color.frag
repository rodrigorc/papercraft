#version 150

uniform vec4 color;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = color;
}
