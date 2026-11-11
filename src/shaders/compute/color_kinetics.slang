// One color step. Temperature is degrees Fahrenheit, the unit stored by the heat solver.
// dt is seconds (HeatParams::timeStep). L* and b* are copied through; only a* changes.
//
// aInf(T_C) = a0 - 15.08 / (1 + 1.84 * exp(0.17 * (47.17 - T_C)))
// k(T_K)    = k0 * exp(-Ea / (R * T_K))
//             k0 (kInf) = 1.4482e10 s^-1, Ea = 80.7397 kJ/mol = 80739.7 J/mol, R = 8.314
// a_next    = aInf + (a - aInf) * exp(-k * dt)
//
// The exponential step is the exact integral while temperature is held constant over dt.
// It is not the forward-Euler update.

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
RWTexture3D<float4> lab0; // immutable; G is a*_0

[[vk::binding(4, 0)]]
[[vk::image_format("rgba16f")]]
RWTexture3D<float4> labOut;

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

    float tF = temperature[i];
    float tC = (tF - 32.0) * (5.0 / 9.0);
    float tK = tC + 273.15;
    if (!(tK > 0.0) || tK > 1.0e4) {
        labOut[id] = current;
        return;
    }

    float a0 = lab0[id].g;
    float aInf = a0 - 15.08 / (1.0 + 1.84 * exp(0.17 * (47.17 - tC)));
    float k = 1.4482e10 * exp(-80739.7 / (8.314 * tK));
    float aNext = aInf + (current.g - aInf) * exp(-k * params.dt);
    labOut[id] = float4(current.r, aNext, current.b, current.a);
}
