#pragma once

#include "color_solver.hpp"
#include "compute_context.hpp"
#include "heat_solver.hpp"
#include "voxelizer.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

struct SteakHeatResult{
    VoxelGrid voxels;  // in model units
    HeatGrid heat;     // same cells as voxels; CELL_SOLID where voxels is nonzero, pan on the low params.panAxis layer
};

// Mesh -> voxelizer -> heat solver. The mesh is voxelized with cells params.spacing meters wide, so
// voxels and heat cells line up one to one. The voxel grid keeps one empty layer of padding on every
// side of the mesh; the low layer along params.panAxis becomes the pan, touching that face of the steak.
// params.dims is ignored (the grid size comes from the mesh bounds).
//
// metersPerUnit converts model coordinates to meters (0.01 for a model in centimeters).
// `progress`, when set, is called with the fraction of the voxelize-and-solve work finished, in [0, 1].
SteakHeatResult simulateMeshHeat(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                                 const std::vector<uint32_t>& indices, float metersPerUnit, HeatParams params,
                                 uint32_t steps, const std::function<void(float)>& progress = {});

// Same heat run, plus a Lab color volume initialized from `rgba` (packed sRGB8) through `texcoords`
// (one UV per position). HeatGrid::color receives linear sRGB. One color step follows each heat step;
// dt is params.timeStep in seconds. Temperature stays in Fahrenheit.
SteakHeatResult simulateMeshHeatColor(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                                      const std::vector<glm::vec2>& texcoords, const std::vector<uint32_t>& indices,
                                      float metersPerUnit, HeatParams params, uint32_t steps, const uint8_t* rgba,
                                      uint32_t textureWidth, uint32_t textureHeight,
                                      const std::function<void(float)>& progress = {});

// simulateMeshHeatColor played live: the voxels, heat field and color volume stay on the GPU, and each
// advance() continues the cook. read() returns the same HeatGrid simulateMeshHeatColor would after that
// many steps. `rgba` is copied to the GPU in the constructor. ctx must outlive the session.
class MeshHeatColorSession{
public:
    MeshHeatColorSession(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                         const std::vector<glm::vec2>& texcoords, const std::vector<uint32_t>& indices, float metersPerUnit,
                         HeatParams params, const uint8_t* rgba, uint32_t textureWidth, uint32_t textureHeight);

    void advance(uint32_t steps){ session->advance(steps); }
    HeatGrid read();
    uint32_t stepsDone() const { return session->stepsDone(); }
    const VoxelGrid& voxels() const { return voxelGrid; }

private:
    VoxelGrid voxelGrid;
    ColorSolver color;
    HeatSolver solver;
    std::optional<HeatSession> session;
};
