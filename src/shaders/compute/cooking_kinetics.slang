// Two-step cooking kinetics for L* and a*, R -k1-> Q -k2-> P with Q = coordinate / raw value.
// Keep the constants and the arithmetic order in step with src/sim/color_model.hpp, which
// documents the model.
//
// Each coordinate's state is (q, c): q = Q/Q_0 and c = fraction of the first reaction's reactant
// left. Rates are s^-1. Below a table's first row nothing changes; above 90 C the 90 C row is used.

struct KineticsParams {
    float n;  // Q_inf / Q_0
    float m;  // (R0 - R_inf) / Q_0
    float k1; // s^-1
    float k2; // s^-1
};

static const uint LIGHTNESS_ROWS = 9;
static const float LIGHTNESS_CELSIUS[9] = { 30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0 };
static const float LIGHTNESS_N[9] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.2684, 1.4320, 1.4244, 1.4172 };
static const float LIGHTNESS_M[9] = { 0.0383, 0.0906, 0.1642, 0.3095, 0.3339, 0.4287, 0.5541, 0.5342, 0.4462 };
static const float LIGHTNESS_K1[9] = { 0.0014, 0.0027, 0.0031, 0.0053, 0.0123, 0.0305, 0.1100, 0.1527, 0.2009 };
static const float LIGHTNESS_K2[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0004, 0.0036, 0.0163, 0.0163 };

static const uint REDNESS_ROWS = 7;
static const float REDNESS_CELSIUS[7] = { 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0 };
static const float REDNESS_N[7] = { 1.0208, 0.8810, 0.8436, 0.7669, 0.7820, 0.7945, 0.7866 };
static const float REDNESS_M[7] = { 0.1885, 0.2467, 1.1803, 3.8473, 4.3628, 3.9168, 5.4333 };
static const float REDNESS_K1[7] = { 0.0027, 0.0029, 0.0017, 0.0037, 0.0041, 0.0042, 0.0050 };
static const float REDNESS_K2[7] = { 0.0027, 0.0029, 0.0050, 0.0328, 0.0451, 0.0484, 0.0735 };

KineticsParams lerpRows(KineticsParams lo, KineticsParams hi, float w) {
    KineticsParams p;
    p.n = lo.n + (hi.n - lo.n) * w;
    p.m = lo.m + (hi.m - lo.m) * w;
    p.k1 = lo.k1 + (hi.k1 - lo.k1) * w;
    p.k2 = lo.k2 + (hi.k2 - lo.k2) * w;
    return p;
}

KineticsParams lightnessRow(uint i) {
    KineticsParams p = { LIGHTNESS_N[i], LIGHTNESS_M[i], LIGHTNESS_K1[i], LIGHTNESS_K2[i] };
    return p;
}

KineticsParams rednessRow(uint i) {
    KineticsParams p = { REDNESS_N[i], REDNESS_M[i], REDNESS_K1[i], REDNESS_K2[i] };
    return p;
}

KineticsParams lightnessParams(float celsius) {
    KineticsParams p = { 1.0, 0.0, 0.0, 0.0 };
    if (!(celsius >= LIGHTNESS_CELSIUS[0])) return p;
    uint i = 0;
    while (i < LIGHTNESS_ROWS - 2 && celsius >= LIGHTNESS_CELSIUS[i + 1]) ++i;
    float w = min((celsius - LIGHTNESS_CELSIUS[i]) / (LIGHTNESS_CELSIUS[i + 1] - LIGHTNESS_CELSIUS[i]), 1.0);
    return lerpRows(lightnessRow(i), lightnessRow(i + 1), w);
}

KineticsParams rednessParams(float celsius) {
    KineticsParams p = { 1.0, 0.0, 0.0, 0.0 };
    if (!(celsius >= REDNESS_CELSIUS[0])) return p;
    uint i = 0;
    while (i < REDNESS_ROWS - 2 && celsius >= REDNESS_CELSIUS[i + 1]) ++i;
    float w = min((celsius - REDNESS_CELSIUS[i]) / (REDNESS_CELSIUS[i + 1] - REDNESS_CELSIUS[i]), 1.0);
    return lerpRows(rednessRow(i), rednessRow(i + 1), w);
}

// Exact integral while the parameters stay constant over dt. state = (q, c).
float2 stepKinetics(float2 state, KineticsParams p, float dt) {
    if (p.k1 == 0.0 && p.k2 == 0.0) return state;
    float e1 = exp(-p.k1 * dt);
    float e2 = exp(-p.k2 * dt);
    // (e1 - e2) / (k2 - k1), which tends to dt * e1 as k2 -> k1.
    float gap = p.k2 - p.k1;
    float phi = abs(gap * dt) < 1.0e-4 ? dt * e1 : (e1 - e2) / gap;
    return float2(p.n + (state.x - p.n) * e2 + p.k1 * p.m * state.y * phi, state.y * e1);
}

// state = (L* q, L* c, a* q, a* c).
float4 stepCook(float4 state, float celsius, float dt) {
    return float4(stepKinetics(state.xy, lightnessParams(celsius), dt), stepKinetics(state.zw, rednessParams(celsius), dt));
}

// Lab after cooking from raw `lab0`; b* is kept, as in the paper. L* is capped at 100.
float3 cookedLab(float3 lab0, float4 state) {
    return float3(min(lab0.x * state.x, 100.0), lab0.y * state.z, lab0.z);
}
