#pragma once

#include "heat_solver.hpp"

#include <cstdint>
#include <functional>
#include <string>

// Prints the temperature up the grid's center column (first `rows` layers from z = 0, the pan).
void printCenterColumn(const HeatGrid& grid, uint32_t rows);

// Reruns the heat model. Called on a background thread. `progress` receives fractions in [0, 1] as that
// run advances. The returned grid replaces the one on screen.
using HeatSimulator = std::function<HeatGrid(const HeatParams& params, uint32_t steps, const std::function<void(float)>& progress)>;

// Triangle mesh in the same model units the voxelizer used. Empty leaves the window on voxel blocks.
// `texcoords` match `positions` and use OpenGL orientation (V up). `rgba` is the albedo, packed sRGB8.
struct HeatSurface{
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    std::vector<glm::vec2> texcoords;
    std::vector<uint8_t> rgba;
    int width = 0;
    int height = 0;
    float metersPerUnit = 0.01f;
};

// Opens an Open3D window that draws every non-air cell of `grid` as a block colored by temperature
// and blocks until it is closed. When `grid.color` holds one linear sRGB sample per cell, C toggles
// solid cells to that cooked color; the pan stays on the temperature key. When `surface` is set, V
// toggles from those blocks to the textured mesh. Cooked color on the mesh samples the voxel field.
// Coordinates are in millimeters. Saved settings are applied by `simulate`
// when the user confirms a reset. When `gridFollowsMesh` is set, the grid dimensions come from the mesh
// and cell size, so those dimension fields stay read-only.
//
// Keys:
//   X / Y / Z  choose the slice axis
//   Up/Down    show a single slice and move it
//   A          show the whole grid again
//   P          toggle the pan cells
//   C          toggle cooked color, when the grid has a color volume
//   I          voxel volume of the texture before cooking
//   V          toggle voxel blocks and the mesh that samples them
void showHeatGrid(const std::string& title, HeatGrid grid, const HeatParams& params, uint32_t steps, HeatSimulator simulate,
                  bool gridFollowsMesh = false, HeatSurface surface = {});
