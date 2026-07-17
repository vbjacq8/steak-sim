#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D u_slice;

vec3 temperature_color(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 cold = vec3(0.05, 0.12, 0.55);
  vec3 warm = vec3(0.95, 0.35, 0.05);
  vec3 hot = vec3(0.98, 0.92, 0.35);
  return t < 0.5 ? mix(cold, warm, t * 2.0) : mix(warm, hot, (t - 0.5) * 2.0);
}

void main() {
  float temp = texture(u_slice, v_uv).r;
  out_color = vec4(temperature_color(temp), 1.0);
}
