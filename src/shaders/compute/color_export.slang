// Reads the current 3D Lab texture and writes linear sRGB for the viewer.
// The Open3D heat window does not sample Vulkan images. Its framebuffer is linear
// (the heat-map colors are linearized with pow(2.2) before upload), so this pass
// stops at linear RGB and does not encode sRGB.

import lab_color;

struct ColorPush {
    uint3 dims;
    float dt;
};

[[vk::push_constant]]
ConstantBuffer<ColorPush> params;

[[vk::binding(0, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture3D<float4> labImage;

[[vk::binding(1, 0)]]
RWStructuredBuffer<float4> outData; // [0, count) Lab, [count, 2*count) linear RGBA

[shader("compute")]
[numthreads(4, 4, 4)]
void colorExportMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= params.dims)) return;

    uint i = id.x + params.dims.x * (id.y + params.dims.y * id.z);
    uint count = params.dims.x * params.dims.y * params.dims.z;
    float4 lab = labImage[id];
    outData[i] = lab;
    float3 rgb = lab.a > 0.5 ? labToLinearSrgb(lab.rgb) : float3(0.0, 0.0, 0.0);
    outData[count + i] = float4(rgb, lab.a);
}
