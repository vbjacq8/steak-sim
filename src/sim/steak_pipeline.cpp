#include "steak_pipeline.hpp"

#include <stdexcept>

SteakHeatResult simulateMeshHeat(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                                 const std::vector<uint32_t>& indices, float metersPerUnit, HeatParams params,
                                 uint32_t steps){
    if (!(metersPerUnit > 0.0f)) throw std::invalid_argument("simulateMeshHeat: metersPerUnit must be positive");

    SteakHeatResult result;
    Voxelizer voxelizer(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, VOXELIZE_SPV);
    result.voxels = voxelizer.voxelize(positions, indices, params.spacing / metersPerUnit);

    params.dims = result.voxels.dims;
    HeatSolver solver(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, FINITE_DIFFERENCE_SPV);
    result.heat = solver.run(params, makePanCells(params.dims, result.voxels.cells), steps);
    return result;
}
