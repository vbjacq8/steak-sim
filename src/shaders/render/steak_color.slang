// Mesh shader for steak_color_demo. Same camera transform as 28_model_loading.slang.
// The fragment shader cooks the sampled texel in place: sRGB texture -> linear RGB -> CIELAB,
// then takes the doneness color (doneness.slang) for one temperature across the whole model, once
// any time has passed. There is no voxel grid.
// The swapchain is sRGB, so the output stays linear and the display encode happens at present.
// The raw Lab is the texel's own.

import lab_color;
import doneness;

struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
};

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 cook; // x = simulated seconds, y = temperature in degrees Fahrenheit
};

ConstantBuffer<UniformBuffer> ubo;

Sampler2D texture;

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    // sRGB image sample is already linear. Do not decode it again.
    float4 sampled = texture.Sample(vertIn.fragTexCoord);
    float3 lab = linearSrgbToLab(sampled.rgb);
    float tF = ubo.cook.y;
    float seconds = ubo.cook.x;
    float tC = (tF - 32.0) * (5.0 / 9.0);
    if (seconds > 0.0) lab = cookedLab(lab, tC);
    return float4(labToLinearSrgb(lab), sampled.a);
}
