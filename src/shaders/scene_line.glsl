#version 140

uniform mat4 m;
// half view size, actually
uniform vec2 view_size;

in vec3 pos_3d;
in vec3 pos_b;
in vec4 color;
// half thickness, actually
in float thick;
in int top;

out vec4 v_color;

void main(void) {
    v_color = color;
    vec4 va = m * vec4(pos_3d, 1.0);
    vec4 vb = m * vec4(pos_b, 1.0);
    va.xy = (va.xy / va.w + vec2(1.0)) * view_size;
    va.z = (va.z - 0.01) / va.w;
    vb.xy = (vb.xy / vb.w + vec2(1.0)) * view_size;
    vb.z = (vb.z - 0.01) / vb.w;

    vec2 vline = normalize(vb.xy - va.xy);
    vec2 nvline = vec2(-vline.y, vline.x) * thick;

    va.xy = (va.xy + nvline) / view_size - vec2(1.0);

    va.xyz *= va.w;
    gl_Position = va;
    if (top == 0)
    {
        gl_Position.z = gl_Position.z * 0.8 + 0.1 * gl_Position.w;
    } else if (top > 0) {
        gl_Position.z = gl_Position.z * 0.1;
    } else {
        gl_Position.z = gl_Position.z * 0.1 + 0.9 * gl_Position.w;
        v_color.rgb = mix(vec3(0.2, 0.2, 0.4), v_color.rgb, 0.25);
    }
}

###

#version 140

in vec4 v_color;
out vec4 out_frag_color;

void main(void) {
    out_frag_color = v_color;
}
