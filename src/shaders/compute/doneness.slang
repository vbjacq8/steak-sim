// Doneness color from the peak temperature a cell has reached. Keep the table and the arithmetic
// order in step with src/sim/color_model.hpp, which documents where the colors come from.
//
// Each key gives a target Lab and how far the raw Lab moves toward it (0 keeps it, 1 replaces it),
// both linear in temperature between keys. Below the first key nothing changes; above the last key
// the last one holds.

import lab_color;

struct DonenessKey {
    float celsius;
    float3 lab;
    float strength;
};

static const uint DONENESS_KEY_COUNT = 7;
static const DonenessKey DONENESS_KEYS[7] = {
    { 40.0, float3(26.7, 33.5, 22.3), 0.0 },
    { 49.0, float3(26.7, 33.5, 22.3), 1.0 },
    { 54.5, float3(35.4, 37.5, 21.6), 1.0 },
    { 65.6, float3(47.8, 28.7, 21.3), 1.0 },
    { 71.3, float3(30.5, 18.1, 20.9), 1.0 },
    { 152.8, float3(33.5, 16.3, 19.8), 1.0 },
    { 163.9, float3(22.4, 13.8, 19.2), 1.0 },
};

// Peak temperature before any heating has been seen.
static const float NO_PEAK_CELSIUS = -273.15;

// xyz = target Lab, w = strength. NaN and anything below the first key give zero strength.
float4 donenessAt(float peakCelsius) {
    if (!(peakCelsius >= DONENESS_KEYS[0].celsius)) return float4(0.0, 0.0, 0.0, 0.0);
    uint i = 0;
    while (i < DONENESS_KEY_COUNT - 2 && peakCelsius >= DONENESS_KEYS[i + 1].celsius) ++i;
    DonenessKey lo = DONENESS_KEYS[i];
    DonenessKey hi = DONENESS_KEYS[i + 1];
    float w = min((peakCelsius - lo.celsius) / (hi.celsius - lo.celsius), 1.0);
    return float4(lo.lab + (hi.lab - lo.lab) * w, lo.strength + (hi.strength - lo.strength) * w);
}

// Share of the texture's lightness pattern kept by the cooked color (kTextureDetail).
static const float TEXTURE_DETAIL = 0.5;

float labLuminance(float3 lab) { return labWhiteY() * labFInv((lab.x + 16.0) / 116.0); }

// w of donenessAt is the cooked fraction. The cooked color is the table color scaled by
// (texel luminance / referenceY)^TEXTURE_DETAIL in linear RGB.
float3 cookedLab(float3 lab0, float peakCelsius, float referenceY) {
    float4 d = donenessAt(peakCelsius);
    if (!(d.w > 0.0)) return lab0;
    float detail = pow(max(labLuminance(lab0), 0.0) / max(referenceY, 1.0e-4), TEXTURE_DETAIL);
    float3 cooked = linearSrgbToLab(clamp(labToLinearSrgb(d.xyz) * detail, 0.0, 1.0));
    return lab0 + (cooked - lab0) * d.w;
}
