#version 150

uniform mat3 m;

in vec2 pos;

void main(void) {
    gl_Position = vec4((m * vec3(pos, 1.0)).xy, 0.0, 1.0);
}

###

#version 150

uniform vec4 color;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = color;
}
