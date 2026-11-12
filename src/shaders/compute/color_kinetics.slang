// One color step. Temperature is degrees Fahrenheit, the unit stored by the heat solver.
// dt is seconds (HeatParams::timeStep). L* and a* change; b* is copied through.
//
// L* and a* follow the two-step kinetics in cooking_kinetics.slang. Their state lives in a 32-bit
// image that this pass updates in place; the cooked Lab is computed from lab0 and written to labOut.
// The step is the exact integral while temperature is held constant over dt, not forward Euler.

import cooking_kinetics;

static const uint CELL_SOLID = 1u;

struct ColorPush {
    uint3 dims;
    float dt;
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
[[vk::image_format("rgba32f")]]
RWTexture3D<float4> cookState; // (L* q, L* c, a* q, a* c)

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
    float4 next = stepCook(cookState[id], tC, params.dt);
    cookState[id] = next;
    labOut[id] = float4(cookedLab(lab0[id].rgb, next), current.a);
}
