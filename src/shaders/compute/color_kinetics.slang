// One color step. Temperature is degrees Fahrenheit, the unit stored by the heat solver.
//
// Each solid cell keeps the hottest temperature it has seen, in Celsius, in a 32-bit image that this
// pass updates in place. Its color is the doneness color for that peak (doneness.slang), computed
// from the raw Lab in lab0 and written to labOut. Cooling never undoes it.

import doneness;

static const uint CELL_SOLID = 1u;

struct ColorPush {
    uint3 dims;
    float dt; // unused here; present so every color pass shares one push-constant layout
};

[[vk::push_constant]]
ConstantBuffer<ColorPush> params;

[[vk::binding(0, 0)]]
StructuredBuffer<float> temperature; // degrees Fahrenheit

[[vk::binding(1, 0)]]
StructuredBuffer<uint> cellKind;

[[vk::binding(2, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture3D<float4> labIn;

[[vk::binding(3, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture3D<float4> lab0; // immutable raw Lab

[[vk::binding(4, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture3D<float4> labOut;

[[vk::binding(5, 0)]]
[[vk::image_format("r32f")]]
RWTexture3D<float> peakCelsius;

[shader("compute")]
[numthreads(4, 4, 4)]
void colorKineticsMain(uint3 id : SV_DispatchThreadID) {
    if (any(id >= params.dims)) return;

    uint i = id.x + params.dims.x * (id.y + params.dims.y * id.z);
    float4 current = labIn[id];
    if (cellKind[i] != CELL_SOLID) {
        labOut[id] = current;
        return;
    }

    float tC = (temperature[i] - 32.0) * (5.0 / 9.0);
    float peak = max(peakCelsius[id], tC);
    peakCelsius[id] = peak;
    labOut[id] = float4(cookedLab(lab0[id].rgb, peak), current.a);
}
