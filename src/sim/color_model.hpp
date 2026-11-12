#pragma once

// CPU reference for the steak color model. Shaders in src/shaders/compute/lab_color.slang and
// doneness.slang use the same constants and the same arithmetic order.
//
// The heat solver stores temperature in degrees Fahrenheit (HeatParams::initialTemp and the values
// written by finitedifference.slang). The doneness table is in Celsius. One color step runs after
// each heat step and records that step's temperature if it is the hottest so far.

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

inline float srgbChannelToLinear(float encoded){
    return encoded <= 0.04045f ? encoded / 12.92f : std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

inline float linearChannelToSrgb(float linear){
    linear = std::clamp(linear, 0.0f, 1.0f);
    return linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
}

inline glm::vec3 srgbToLinear(glm::vec3 encoded){
    return {srgbChannelToLinear(encoded.x), srgbChannelToLinear(encoded.y), srgbChannelToLinear(encoded.z)};
}

inline glm::vec3 linearToSrgb(glm::vec3 linear){
    return {linearChannelToSrgb(linear.x), linearChannelToSrgb(linear.y), linearChannelToSrgb(linear.z)};
}

// Linear sRGB (D65) to CIE XYZ. The reference white is the sum of each matrix row, so linear
// (1, 1, 1) is Lab (100, 0, 0) in the same float arithmetic the shader uses.
inline float labWhiteX(){ return 0.4124564f + 0.3575761f + 0.1804375f; }
inline float labWhiteY(){ return 0.2126729f + 0.7151522f + 0.0721750f; }
inline float labWhiteZ(){ return 0.0193339f + 0.1191920f + 0.9503041f; }

inline float labF(float t){
    constexpr float eps = 216.0f / 24389.0f;
    constexpr float kappa = 24389.0f / 27.0f;
    return t > eps ? std::pow(t, 1.0f / 3.0f) : (kappa * t + 16.0f) / 116.0f;
}

inline float labFInv(float t){
    constexpr float eps = 216.0f / 24389.0f;
    constexpr float kappa = 24389.0f / 27.0f;
    const float cubed = t * t * t;
    return cubed > eps ? cubed : (116.0f * t - 16.0f) / kappa;
}

inline glm::vec3 linearSrgbToLab(glm::vec3 rgb){
    const float X = 0.4124564f * rgb.x + 0.3575761f * rgb.y + 0.1804375f * rgb.z;
    const float Y = 0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f * rgb.z;
    const float Z = 0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z;
    const float fx = labF(X / labWhiteX());
    const float fy = labF(Y / labWhiteY());
    const float fz = labF(Z / labWhiteZ());
    return {116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz)};
}

// Display conversion. Out-of-gamut channels are clamped. Open3D's heat window consumes linear
// values (it linearizes the heat-map stops itself), so the viewer uses this linear result and
// does not apply a second sRGB encode.
inline glm::vec3 labToLinearSrgb(glm::vec3 lab){
    const float fy = (lab.x + 16.0f) / 116.0f;
    const float fx = fy + lab.y / 500.0f;
    const float fz = fy - lab.z / 200.0f;
    const float X = labWhiteX() * labFInv(fx);
    const float Y = labWhiteY() * labFInv(fy);
    const float Z = labWhiteZ() * labFInv(fz);
    glm::vec3 rgb;
    rgb.x = 3.2404542f * X + -1.5371385f * Y + -0.4985314f * Z;
    rgb.y = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    rgb.z = 0.0556434f * X + -0.2040259f * Y + 1.0572252f * Z;
    return glm::clamp(rgb, 0.0f, 1.0f);
}

inline float fahrenheitToCelsius(float fahrenheit){ return (fahrenheit - 32.0f) * (5.0f / 9.0f); }

// Doneness color from the peak temperature a cell has reached. Cooking does not reverse, so cooling
// keeps the color. Each key gives a target Lab and how far the texel's raw Lab moves toward it;
// both are linear in temperature between keys. Below the first key nothing changes; above the last
// key the last one holds.
//
// Calibrated from a photo of steaks seared to known center temperatures (rare 120 F, medium-rare
// 130 F, medium-well 150 F, well-done 160 F). Center keys are the sampled center colors at those
// temperatures. Crust keys sit at the surface peak that tools/doneness_calibrate finds for the
// rare and well-done centers (30 mm steak, one flip); their colors are a straight-line fit of the
// four sampled crusts against those surface peaks, since single samples vary with the lighting.
struct DonenessKey{
    float celsius;
    glm::vec3 lab;
    float strength;  // 0 keeps the texel, 1 replaces it with `lab`
};

// The 40 C key repeats the rare color so that the blend starts from zero strength without
// drifting toward another color.
inline constexpr int kDonenessKeyCount = 7;
inline constexpr DonenessKey kDonenessKeys[kDonenessKeyCount] = {
    {40.0f, {26.7f, 33.5f, 22.3f}, 0.0f},
    {49.0f, {26.7f, 33.5f, 22.3f}, 1.0f},   // rare center, sRGB (114, 37, 31)
    {54.5f, {35.4f, 37.5f, 21.6f}, 1.0f},   // medium-rare center, (143, 54, 51)
    {65.6f, {47.8f, 28.7f, 21.3f}, 1.0f},   // medium-well center, (167, 93, 79)
    {71.3f, {30.5f, 18.1f, 20.9f}, 1.0f},   // well-done center, (106, 60, 40)
    {152.8f, {33.5f, 16.3f, 19.8f}, 1.0f},  // crust fit at the rare surface peak, (111, 68, 48)
    {163.9f, {22.4f, 13.8f, 19.2f}, 1.0f},  // crust fit at the well-done surface peak, (79, 45, 25)
};

// Peak temperature before any heating has been seen. Below every key, so it leaves color alone.
inline constexpr float kNoPeakCelsius = -273.15f;

struct Doneness{
    glm::vec3 lab{0.0f};
    float strength = 0.0f;
};

// NaN and anything below the first key give zero strength.
inline Doneness donenessAt(float peakCelsius){
    Doneness d;
    if (!(peakCelsius >= kDonenessKeys[0].celsius)) return d;
    int i = 0;
    while (i < kDonenessKeyCount - 2 && peakCelsius >= kDonenessKeys[i + 1].celsius) ++i;
    const DonenessKey& lo = kDonenessKeys[i];
    const DonenessKey& hi = kDonenessKeys[i + 1];
    const float w = std::min((peakCelsius - lo.celsius) / (hi.celsius - lo.celsius), 1.0f);
    d.lab = lo.lab + (hi.lab - lo.lab) * w;
    d.strength = lo.strength + (hi.strength - lo.strength) * w;
    return d;
}

// Lab of a texel with raw Lab `lab0` once its cell has peaked at `peakCelsius`.
inline glm::vec3 cookedLab(glm::vec3 lab0, float peakCelsius){
    const Doneness d = donenessAt(peakCelsius);
    return lab0 + (d.lab - lab0) * d.strength;
}

// Peak update after a heat step. `fahrenheit` is the heat-solver temperature; NaN keeps the peak.
inline float raisePeak(float peakCelsius, float fahrenheit){
    return std::max(peakCelsius, fahrenheitToCelsius(fahrenheit));
}
