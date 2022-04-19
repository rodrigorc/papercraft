#version 140

/*
A single triangle can cover the whole viewport:
 3 *
   | \
 2 +  +
   |    \
 1 +-----+
   |     | \
 0 +  +  |  +
   |     |    \
-1 *--+--+--+--*
  -1  0  1  2  3
*/

vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
);

void main(void) {
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}

###

#version 140

uniform vec4 color;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = color;
}
