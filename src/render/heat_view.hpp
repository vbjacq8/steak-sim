#pragma once

#include "heat_solver.hpp"

#include <cstdint>
#include <functional>
#include <string>

// Prints the temperature up the grid's center column (first `rows` layers from y = 0).
void printCenterColumn(const HeatGrid& grid, uint32_t rows);

// Reruns the heat model. Called on a background thread. `progress` receives fractions in [0, 1] as that
// run advances. The returned grid replaces the one on screen.
using HeatSimulator = std::function<HeatGrid(const HeatParams& params, uint32_t steps, const std::function<void(float)>& progress)>;

// Opens an Open3D window that draws every non-air cell of `grid` as a block colored by temperature
// and blocks until it is closed. Coordinates are in millimeters. Saved settings are applied by `simulate`
// when the user confirms a reset. When `gridFollowsMesh` is set, the grid dimensions come from the mesh
// and cell size, so those dimension fields stay read-only.
//
// Keys:
//   X / Y / Z  choose the slice axis
//   Up/Down    show a single slice and move it
//   A          show the whole grid again
//   P          toggle the pan cells
void showHeatGrid(const std::string& title, HeatGrid grid, const HeatParams& params, uint32_t steps, HeatSimulator simulate,
                  bool gridFollowsMesh = false);
