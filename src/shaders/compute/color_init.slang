// Writes the initial 3D Lab texture from the steak's 2D sRGB texture.
//
// `voxelUv` comes from the voxelizer: occupied voxels hold the barycentric UV of the closest
// point on the mesh, empty voxels hold (-1, -1). Only CELL_SOLID voxels become meat. Sampling
// an sRGB image returns linear RGB, which is then converted to CIELAB. The same value is stored
// in the immutable lab0 image and in the current Lab image. The peak temperature starts below every
// doneness key, so nothing cooks until a heat step is seen.

import doneness;
import lab_color;

static const uint CELL_SOLID = 1u; // must match heat_solver.hpp

struct ColorPush {
    uint3 dims;
    float referenceY; // unused here; present so every color pass shares one push-constant layout
};

[[vk::push_constant]]
ConstantBuffer<ColorPush> params;

[[vk::binding(0, 0)]]
StructuredBuffer<uint> cellKind;

[[vk::binding(1, 0)]]
StructuredBuffer<float2> voxelUv;

[[vk::binding(2, 0)]]
Texture2D<float4> albedo;

[[vk::binding(3, 0)]]
SamplerState albedoSampler;

[[vk::binding(4, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture3D<float4> lab0;

[[vk::binding(5, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture3D<float4> labCurrent;

[[vk::binding(6, 0)]]
[[vk::image_format("r32f")]]
RWTexture3D<float> peakCelsius;

[shader("compute")]
[numthreads(4, 4, 4)]
void colorInitMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= params.dims)) return;

    uint i = id.x + params.dims.x * (id.y + params.dims.y * id.z);
    float4 lab = float4(0.0, 0.0, 0.0, 0.0);
    if (cellKind[i] == CELL_SOLID) {
        // SampleLevel: compute shaders have no screen-space derivatives.
        float3 linearRgb = albedo.SampleLevel(albedoSampler, voxelUv[i], 0.0).rgb;
        lab = float4(linearSrgbToLab(linearRgb), 1.0);
    }
    lab0[id] = lab;
    labCurrent[id] = lab;
    peakCelsius[id] = NO_PEAK_CELSIUS;
}
