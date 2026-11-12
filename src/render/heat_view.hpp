#pragma once

#include "heat_solver.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// Prints the temperature up the grid's center column, the first `rows` layers from the pan along HeatGrid::panAxis.
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

// A simulation the live window plays. The window calls it from one worker thread, one call at a time.
class LiveHeat{
public:
    virtual ~LiveHeat() = default;
    // Drops the run in progress and sets up `params` at time zero. Returns that starting field.
    virtual HeatGrid restart(const HeatParams& params) = 0;
    // Continues the run by `steps` steps of params.timeStep and returns the field at the new time.
    virtual HeatGrid advance(uint32_t steps) = 0;
};

// The showHeatGrid window with a cooking clock in place of the reset button. It opens on `params` at
// time zero and stays there until Start; settings typed before then are what Start cooks with. While
// playing, the clock advances `speed` simulated seconds per real second and the view redraws as the
// field changes, until the clock reaches HeatParams::duration. Pause holds the clock; Reset returns to
// time zero with the current settings. Settings typed mid-cook wait for the next Start or Reset.
//
// Extra keys:
//   Space      start, pause or resume
//   [ / ]      slower / faster
void showLiveHeatGrid(const std::string& title, const HeatParams& params, std::unique_ptr<LiveHeat> live,
                      bool gridFollowsMesh = false, HeatSurface surface = {});
