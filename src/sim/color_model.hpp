#pragma once

// CPU reference for the steak color model. Shaders in src/shaders/compute/lab_color.slang and
// cooking_kinetics.slang use the same constants and the same arithmetic order.
//
// The heat solver stores temperature in degrees Fahrenheit (HeatParams::initialTemp and the values
// written by finitedifference.slang). The kinetics tables are in Celsius. The kinetics
// timestep is seconds, matching HeatParams::timeStep. One color step runs after each heat step and
// treats that step's temperature as constant over the interval.

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

// Lightness L* and redness a* each follow two successive first-order reactions, R -k1-> Q -k2-> P,
// where Q is the coordinate over its raw value (Moya et al. 2021, Eqs. 12-13, Tables 3-4). Per unit
// of the raw value, at constant temperature:
//   dr/dt = -k1 r,                       r(0) = m
//   dq/dt = -k2 (q - n) + k1 r,          q(0) = 1
// with n = Q_inf / Q_0 and m = (R0 - R_inf) / Q_0. While temperature varies, r = m(T) * c, where
// c is the fraction of reactant left (c(0) = 1), so each coordinate's state is (q, c).
//
// The tables label k1 and k2 min^-1, but they are applied here as s^-1: in minutes a* would still
// rise after 30 min at 90 C, while the paper's data rise briefly and then fall toward n.
// Below a table's first row nothing changes; above 90 C the 90 C row is used.
struct KineticsParams{
    float n = 1.0f;   // Q_inf / Q_0
    float m = 0.0f;   // (R0 - R_inf) / Q_0
    float k1 = 0.0f;  // s^-1
    float k2 = 0.0f;  // s^-1
};

struct KineticsState{
    float ratio = 1.0f;     // q = Q / Q_0
    float reactant = 1.0f;  // c, fraction of the first reaction's reactant left
};

template <int Rows>
struct KineticsTable{
    float celsius[Rows];
    float n[Rows];
    float m[Rows];
    float k1[Rows];
    float k2[Rows];
};

// Table 3, L*. Up to 55 C the fit has no second reaction (n = 1, k2 = 0).
inline constexpr KineticsTable<9> kLightnessTable{
    {30.0f, 40.0f, 45.0f, 50.0f, 55.0f, 60.0f, 70.0f, 80.0f, 90.0f},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.2684f, 1.4320f, 1.4244f, 1.4172f},
    {0.0383f, 0.0906f, 0.1642f, 0.3095f, 0.3339f, 0.4287f, 0.5541f, 0.5342f, 0.4462f},
    {0.0014f, 0.0027f, 0.0031f, 0.0053f, 0.0123f, 0.0305f, 0.1100f, 0.1527f, 0.2009f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0004f, 0.0036f, 0.0163f, 0.0163f},
};

// Table 4, a*.
inline constexpr KineticsTable<7> kRednessTable{
    {45.0f, 50.0f, 55.0f, 60.0f, 70.0f, 80.0f, 90.0f},
    {1.0208f, 0.8810f, 0.8436f, 0.7669f, 0.7820f, 0.7945f, 0.7866f},
    {0.1885f, 0.2467f, 1.1803f, 3.8473f, 4.3628f, 3.9168f, 5.4333f},
    {0.0027f, 0.0029f, 0.0017f, 0.0037f, 0.0041f, 0.0042f, 0.0050f},
    {0.0027f, 0.0029f, 0.0050f, 0.0328f, 0.0451f, 0.0484f, 0.0735f},
};

// Linear in temperature between table rows. NaN and anything below the first row give zero rates.
template <int Rows>
KineticsParams kineticsParams(const KineticsTable<Rows>& t, float celsius){
    KineticsParams p;
    if (!(celsius >= t.celsius[0])) return p;
    int i = 0;
    while (i < Rows - 2 && celsius >= t.celsius[i + 1]) ++i;
    const float w = std::min((celsius - t.celsius[i]) / (t.celsius[i + 1] - t.celsius[i]), 1.0f);
    p.n = t.n[i] + (t.n[i + 1] - t.n[i]) * w;
    p.m = t.m[i] + (t.m[i + 1] - t.m[i]) * w;
    p.k1 = t.k1[i] + (t.k1[i + 1] - t.k1[i]) * w;
    p.k2 = t.k2[i] + (t.k2[i + 1] - t.k2[i]) * w;
    return p;
}

inline KineticsParams lightnessParams(float celsius){ return kineticsParams(kLightnessTable, celsius); }
inline KineticsParams rednessParams(float celsius){ return kineticsParams(kRednessTable, celsius); }

// Exact integral of the system above while the parameters stay constant over dtSeconds.
inline KineticsState stepKinetics(KineticsState s, const KineticsParams& p, float dtSeconds){
    if (p.k1 == 0.0f && p.k2 == 0.0f) return s;
    const float e1 = std::exp(-p.k1 * dtSeconds);
    const float e2 = std::exp(-p.k2 * dtSeconds);
    // (e1 - e2) / (k2 - k1), which tends to dt * e1 as k2 -> k1.
    const float gap = p.k2 - p.k1;
    const float phi = std::abs(gap * dtSeconds) < 1.0e-4f ? dtSeconds * e1 : (e1 - e2) / gap;
    return {p.n + (s.ratio - p.n) * e2 + p.k1 * p.m * s.reactant * phi, s.reactant * e1};
}

// Q / Q_0 after `seconds` at a constant temperature, starting raw (Eqs. 12-13).
inline float lightnessRatioAt(float celsius, float seconds){
    return stepKinetics(KineticsState{}, lightnessParams(celsius), seconds).ratio;
}
inline float rednessRatioAt(float celsius, float seconds){
    return stepKinetics(KineticsState{}, rednessParams(celsius), seconds).ratio;
}

struct CookState{
    KineticsState lightness;
    KineticsState redness;
};

// One piecewise-constant step of the voxel color. `fahrenheit` is the heat-solver temperature.
inline CookState stepCook(CookState s, float fahrenheit, float dtSeconds){
    const float celsius = fahrenheitToCelsius(fahrenheit);
    return {stepKinetics(s.lightness, lightnessParams(celsius), dtSeconds),
            stepKinetics(s.redness, rednessParams(celsius), dtSeconds)};
}

// Lab after cooking from raw `lab0`, given (L*/L*_0, a*/a*_0); b* is kept, as in the paper.
// L* is capped at 100.
inline glm::vec3 cookedLab(glm::vec3 lab0, glm::vec2 ratio){
    return {std::min(lab0.x * ratio.x, 100.0f), lab0.y * ratio.y, lab0.z};
}

inline glm::vec3 cookedLab(glm::vec3 lab0, const CookState& s){
    return cookedLab(lab0, glm::vec2(s.lightness.ratio, s.redness.ratio));
}
