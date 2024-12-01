#version 140

uniform mat4 m;
uniform mat3 mnormal;
uniform vec3 lights[2];

in vec3 pos_3d;
in vec3 normal;
in vec2 uv;
in float mat;
in vec4 color;
in int top;

out vec3 v_uv;
out float v_light;
out vec4 v_color;

void main(void) {
    gl_Position = m * vec4(pos_3d, 1.0);
    vec3 obj_normal = normalize(mnormal * normal);

    float light = 0.2;
    for (int i = 0; i < 2; ++i) {
        float diffuse = max(abs(dot(obj_normal, -lights[i])), 0.0);
        light += diffuse;
    }
    v_light = light;
    v_uv = vec3(uv, mat);
    v_color = color;
    if (top != 0) {
        gl_Position.z = gl_Position.z * 0.1;
    } else {
        gl_Position.z = gl_Position.z * 0.9 + 0.1;
    }
}

###

#version 140

uniform bool texturize;
uniform sampler2DArray tex;

in vec3 v_uv;
in float v_light;
in vec4 v_color;
out vec4 out_frag_color;

void main(void) {
    vec4 base;

    if (gl_FrontFacing)
    {
        vec4 c;
        if (texturize) {
            c = texture(tex, v_uv);
        } else {
            c = vec4(0.75, 0.75, 0.75, 1.0);
        }
        base = mix(c, vec4(v_color.rgb, 1.0), v_color.a);
    }
    else
    {
        base = mix(vec4(0.8, 0.3, 0.3, 1.0), vec4(v_color.rgb, 1.0), v_color.a / 2.0);
    }
    // do alpha blending with full-white and output a fully opaque fragment, simulating the texture over paper
    out_frag_color = vec4(v_light * mix(vec3(1.0, 1.0, 1.0), base.rgb, base.a), 1.0);
}
