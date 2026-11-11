#pragma once

// CPU reference for the steak color model. Shaders in src/shaders/compute/lab_color.slang and
// color_kinetics.slang use the same constants and the same arithmetic order.
//
// The heat solver stores temperature in degrees Fahrenheit (HeatParams::initialTemp and the values
// written by finitedifference.slang). Arrhenius uses Kelvin; the equilibrium sigmoid uses Celsius.
// The kinetics timestep is seconds, matching HeatParams::timeStep. One color step runs after each
// heat step and treats that step's temperature as constant over the interval.

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

inline float equilibriumA(float a0, float celsius){
    const float width = std::exp(0.17f * (47.17f - celsius));
    return a0 - 15.08f / (1.0f + 1.84f * width);
}

// Paper fit used for now: Ea = 80.7397 kJ/mol, k0 (kInf) = 1.4482e10 s^-1.
// Ea is 80739.7 J/mol so it matches R = 8.314 J/(mol·K).
inline float arrheniusK(float kelvin){ return 1.4482e10f * std::exp(-80739.7f / (8.314f * kelvin)); }

// Exact integral of da/dt = -k (a - aInf) while k and aInf stay constant over dtSeconds.
inline float integrateA(float a, float aInf, float k, float dtSeconds){
    return aInf + (a - aInf) * std::exp(-k * dtSeconds);
}

// One piecewise-constant step. `fahrenheit` is the heat-solver temperature. Non-positive or
// absurd Kelvin values leave `a` unchanged.
inline float stepLabA(float a, float a0, float fahrenheit, float dtSeconds){
    const float kelvin = fahrenheitToCelsius(fahrenheit) + 273.15f;
    if (!(kelvin > 0.0f) || kelvin > 1.0e4f) return a;
    const float aInf = equilibriumA(a0, fahrenheitToCelsius(fahrenheit));
    return integrateA(a, aInf, arrheniusK(kelvin), dtSeconds);
}
