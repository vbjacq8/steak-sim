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

// Fat/lean ramp as fractions of the reference a* (kFatRedness, kLeanRedness), and the peak range where
// crust browning takes over from the center colors (kBrowningStartCelsius, kBrowningFullCelsius).
static const float FAT_REDNESS = 0.25;
static const float LEAN_REDNESS = 0.6;
static const float BROWNING_START_CELSIUS = 71.3;
static const float BROWNING_FULL_CELSIUS = 152.8;

float meatiness(float a0, float referenceA) {
    if (!(referenceA > 1.0)) return 1.0;
    return clamp((a0 / referenceA - FAT_REDNESS) / (LEAN_REDNESS - FAT_REDNESS), 0.0, 1.0);
}

float browning(float peakCelsius) {
    return clamp((peakCelsius - BROWNING_START_CELSIUS) / (BROWNING_FULL_CELSIUS - BROWNING_START_CELSIUS), 0.0, 1.0);
}

// Fat table (kFatKeys in color_model.hpp).
static const uint FAT_KEY_COUNT = 4;
static const DonenessKey FAT_KEYS[4] = {
    { 50.0, float3(64.9, 11.7, 23.6), 0.0 },
    { 57.2, float3(64.9, 11.7, 23.6), 0.85 },
    { 62.8, float3(68.3, 10.6, 36.3), 0.85 },
    { 73.9, float3(63.5, 4.6, 18.1), 0.85 },
};

// Same interpolation as donenessAt, over FAT_KEYS.
float4 fatDonenessAt(float peakCelsius) {
    if (!(peakCelsius >= FAT_KEYS[0].celsius)) return float4(0.0, 0.0, 0.0, 0.0);
    uint i = 0;
    while (i < FAT_KEY_COUNT - 2 && peakCelsius >= FAT_KEYS[i + 1].celsius) ++i;
    DonenessKey lo = FAT_KEYS[i];
    DonenessKey hi = FAT_KEYS[i + 1];
    float w = min((peakCelsius - lo.celsius) / (hi.celsius - lo.celsius), 1.0);
    return float4(lo.lab + (hi.lab - lo.lab) * w, lo.strength + (hi.strength - lo.strength) * w);
}

// Fat first follows the fat table. Then w of donenessAt, the cooked fraction, applies; fat takes only
// its browning share. The cooked color is the table color scaled by (texel luminance / referenceY)^
// TEXTURE_DETAIL in linear RGB. referenceY and referenceA are the median luminance and a* of the meat.
float3 cookedLab(float3 lab0, float peakCelsius, float referenceY, float referenceA) {
    float meat = meatiness(lab0.y, referenceA);
    float4 fat = fatDonenessAt(peakCelsius);
    float3 base = lab0 + (fat.xyz - lab0) * ((1.0 - meat) * fat.w);
    float4 d = donenessAt(peakCelsius);
    float strength = d.w * (meat + (1.0 - meat) * browning(peakCelsius));
    if (!(strength > 0.0)) return base;
    float detail = pow(max(labLuminance(lab0), 0.0) / max(referenceY, 1.0e-4), TEXTURE_DETAIL);
    float3 cooked = linearSrgbToLab(clamp(labToLinearSrgb(d.xyz) * detail, 0.0, 1.0));
    return base + (cooked - base) * strength;
}
