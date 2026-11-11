// Linear sRGB (D65) <-> CIELAB. Keep the arithmetic in the same order as src/sim/color_model.hpp.
// The 2D steak texture is an sRGB Vulkan image, so sampling it already returns linear RGB.
// Do not run the sRGB transfer function again on a sampled texel.

static const float LAB_EPS = 216.0 / 24389.0;
static const float LAB_KAPPA = 24389.0 / 27.0;

float labWhiteX() { return 0.4124564 + 0.3575761 + 0.1804375; }
float labWhiteY() { return 0.2126729 + 0.7151522 + 0.0721750; }
float labWhiteZ() { return 0.0193339 + 0.1191920 + 0.9503041; }

float labF(float t) {
    if (t > LAB_EPS) return pow(t, 1.0 / 3.0);
    return (LAB_KAPPA * t + 16.0) / 116.0;
}

float labFInv(float t) {
    float cubed = t * t * t;
    if (cubed > LAB_EPS) return cubed;
    return (116.0 * t - 16.0) / LAB_KAPPA;
}

float3 linearSrgbToLab(float3 rgb) {
    float X = 0.4124564 * rgb.x + 0.3575761 * rgb.y + 0.1804375 * rgb.z;
    float Y = 0.2126729 * rgb.x + 0.7151522 * rgb.y + 0.0721750 * rgb.z;
    float Z = 0.0193339 * rgb.x + 0.1191920 * rgb.y + 0.9503041 * rgb.z;
    float fx = labF(X / labWhiteX());
    float fy = labF(Y / labWhiteY());
    float fz = labF(Z / labWhiteZ());
    return float3(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz));
}

float3 labToLinearSrgb(float3 lab) {
    float fy = (lab.x + 16.0) / 116.0;
    float fx = fy + lab.y / 500.0;
    float fz = fy - lab.z / 200.0;
    float X = labWhiteX() * labFInv(fx);
    float Y = labWhiteY() * labFInv(fy);
    float Z = labWhiteZ() * labFInv(fz);
    float3 rgb = float3(
        3.2404542 * X + -1.5371385 * Y + -0.4985314 * Z,
        -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z,
        0.0556434 * X + -0.2040259 * Y + 1.0572252 * Z);
    return clamp(rgb, 0.0, 1.0);
}
