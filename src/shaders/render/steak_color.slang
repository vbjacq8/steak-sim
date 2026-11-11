// Mesh shader for steak_color_demo. Same camera transform as 28_model_loading.slang.
// The fragment shader cooks the sampled texel in place: sRGB texture -> linear RGB -> CIELAB,
// then a* follows the paper kinetics at one temperature for the whole model. There is no voxel grid.
// The swapchain is sRGB, so the output stays linear and the display encode happens at present.
// L* and b* stay at the texture's original values. a*_0 is the texel's own a*.

import lab_color;

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
    float a0 = lab.y;

    float tF = ubo.cook.y;
    float seconds = ubo.cook.x;
    float tC = (tF - 32.0) * (5.0 / 9.0);
    float tK = tC + 273.15;
    if (tK > 0.0 && tK <= 1.0e4 && seconds > 0.0) {
        // Constant temperature, so the stepwise exponential update is this closed form.
        // k0 (kInf) = 1.4482e10 s^-1, Ea = 80.7397 kJ/mol = 80739.7 J/mol, R = 8.314.
        float aInf = a0 - 15.08 / (1.0 + 1.84 * exp(0.17 * (47.17 - tC)));
        float k = 1.4482e10 * exp(-80739.7 / (8.314 * tK));
        lab.y = aInf + (a0 - aInf) * exp(-k * seconds);
    }
    return float4(labToLinearSrgb(lab), sampled.a);
}
