#pragma once

#include "heat_solver.hpp"

#include <cstdint>
#include <string>

// Prints the temperature up the grid's center column (first `rows` layers from y = 0).
void printCenterColumn(const HeatGrid& grid, uint32_t rows);

// Opens an Open3D window that draws every non-air cell of `grid` as a block colored by temperature
// and blocks until it is closed. Coordinates are in millimeters.
//
// Keys:
//   X / Y / Z  choose the slice axis
//   Up/Down    show a single slice and move it
//   A          show the whole grid again
//   P          toggle the pan cells
void showHeatGrid(const std::string& title, HeatGrid grid, const HeatParams& params, uint32_t steps);
