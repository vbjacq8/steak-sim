#pragma once

#include "compute_context.hpp"
#include "heat_solver.hpp"
#include "voxelizer.hpp"

#include <cstdint>
#include <functional>
#include <vector>

struct SteakHeatResult{
    VoxelGrid voxels;  // in model units
    HeatGrid heat;     // same cells as voxels; CELL_SOLID where voxels is nonzero, pan at y = 0
};

// Mesh -> voxelizer -> heat solver. The mesh is voxelized with cells params.spacing meters wide, so
// voxels and heat cells line up one to one. The voxel grid keeps one empty layer of padding under
// the mesh; that y = 0 layer becomes the pan, touching the bottom of the steak. params.dims is ignored
// (the grid size comes from the mesh bounds).
//
// metersPerUnit converts model coordinates to meters (0.01 for a model in centimeters).
// `progress`, when set, is called with the fraction of the voxelize-and-solve work finished, in [0, 1].
SteakHeatResult simulateMeshHeat(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                                 const std::vector<uint32_t>& indices, float metersPerUnit, HeatParams params,
                                 uint32_t steps, const std::function<void(float)>& progress = {});
