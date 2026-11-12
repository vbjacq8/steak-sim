#pragma once

// Steak model preparation shared by steak_heat_demo and steak_heat_live.
//
// A glTF with several base-color images, such as src/assets/steak2/*.glb, packs those images into one
// atlas so the fat cap and the lean each keep their own texture. A model that is not already about
// the size of a steak (its long axis, read as centimeters, outside 5–45 cm) is scaled so that axis is
// 22 cm. Overlapping duplicate shells are dropped. Side walls whose texture coordinates collapse to a
// line (a top-down projection) are given coordinates that wrap around the steak instead. The pan goes
// against the model's thinnest axis: y for steak1 (which stands on its edge in z), z for steak2.

#include "heat_view.hpp"
#include "mesh_loader.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct RgbaImage{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

struct DemoMesh{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> solverUvs;   // V down, matching the atlas rows, for the voxelizer
    std::vector<glm::vec2> displayUvs;  // V up, for the Open3D window
    std::vector<uint32_t> indices;
    RgbaImage atlas;
    float metersPerUnit = 0.01f;
    uint32_t panAxis = 2;  // the model's thinnest axis; the demos put the pan against its low side
};

float longestAxis(const std::vector<glm::vec3>& positions);

// `fallbackTexture` is used when the model carries no base-color images (an OBJ).
DemoMesh prepareDemoMesh(const Mesh& source, const std::string& fallbackTexture);

// The textured mesh the heat window draws when the voxel volume is switched off.
HeatSurface demoSurface(const DemoMesh& mesh);

// Throws std::invalid_argument when cells `spacing` meters wide would put over 500 on the model's long axis.
void checkCellCount(const DemoMesh& mesh, float spacing);
