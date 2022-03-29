#version 150

uniform mat4 m;
uniform mat3 mnormal;

uniform vec3 lights[2];
in vec3 pos;
in vec3 normal;
in vec2 uv;
in vec4 status;

out vec2 v_uv;
out float v_light;
out vec4 v_status;

void main(void) {
    gl_Position = m * vec4(pos, 1.0);
    vec3 obj_normal = normalize(mnormal * normal);

    float light = 0.2;
    for (int i = 0; i < 2; ++i) {
        float diffuse = max(abs(dot(obj_normal, -lights[i])), 0.0);
        light += diffuse;
    }
    v_light = light;
    v_uv = uv;
    v_status = status;
    if (status.a != 0.0) {
        gl_Position.z = 0.0;
    }
}

###

#version 150

uniform sampler2D tex;
uniform vec4 color;

in vec2 v_uv;
in float v_light;
in vec4 v_status;
out vec4 out_frag_color;

void main(void) {
    vec4 base;

    if (gl_FrontFacing)
    {
        base = mix(texture2D(tex, v_uv), vec4(v_status.rgb, 1.0), v_status.a);
    }
    else
    {
        base = mix(vec4(0.8, 0.3, 0.3, 1.0), vec4(v_status.rgb, 1.0), v_status.a / 2.0);
    }
    out_frag_color = vec4(v_light * base.rgb, base.a);
}
