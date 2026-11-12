#include "steak_pipeline.hpp"

#include "color_solver.hpp"

#include <stdexcept>

SteakHeatResult simulateMeshHeat(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                                 const std::vector<uint32_t>& indices, float metersPerUnit, HeatParams params,
                                 uint32_t steps, const std::function<void(float)>& progress){
    if (!(metersPerUnit > 0.0f)) throw std::invalid_argument("simulateMeshHeat: metersPerUnit must be positive");

    if (progress) progress(0.02f);
    SteakHeatResult result;
    Voxelizer voxelizer(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, VOXELIZE_SPV);
    result.voxels = voxelizer.voxelize(positions, indices, params.spacing / metersPerUnit);
    if (progress) progress(0.2f);

    params.dims = result.voxels.dims;
    HeatSolver solver(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, FINITE_DIFFERENCE_SPV);
    const std::vector<uint32_t> cells = makePanCells(params.dims, result.voxels.cells);
    if (progress){
        result.heat = solver.run(params, cells, steps, [&](float fraction){ progress(0.2f + 0.8f * fraction); });
    } else{
        result.heat = solver.run(params, cells, steps);
    }
    return result;
}

SteakHeatResult simulateMeshHeatColor(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                                      const std::vector<glm::vec2>& texcoords, const std::vector<uint32_t>& indices,
                                      float metersPerUnit, HeatParams params, uint32_t steps, const uint8_t* rgba,
                                      uint32_t textureWidth, uint32_t textureHeight,
                                      const std::function<void(float)>& progress){
    if (!(metersPerUnit > 0.0f)) throw std::invalid_argument("simulateMeshHeatColor: metersPerUnit must be positive");

    if (progress) progress(0.02f);
    SteakHeatResult result;
    Voxelizer voxelizer(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, VOXELIZE_SPV);
    result.voxels = voxelizer.voxelize(positions, texcoords, indices, params.spacing / metersPerUnit);
    if (progress) progress(0.2f);

    params.dims = result.voxels.dims;
    const std::vector<uint32_t> cells = makePanCells(params.dims, result.voxels.cells);
    ColorSolver color(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, COLOR_INIT_SPV, COLOR_KINETICS_SPV,
                      COLOR_EXPORT_SPV);
    color.prepareGrid(result.voxels.dims, cells, result.voxels.uv, rgba, textureWidth, textureHeight, params.timeStep);

    HeatSolver solver(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, FINITE_DIFFERENCE_SPV);
    if (progress){
        result.heat = solver.run(params, cells, steps, [&](float fraction){ progress(0.2f + 0.8f * fraction); }, &color);
    } else{
        result.heat = solver.run(params, cells, steps, {}, &color);
    }
    color.download();
    result.heat.color = color.linearRgb();
    result.heat.initialColor = color.initialLinearRgb();
    result.heat.cookRatio = color.cookRatio();
    result.heat.sampleUv = result.voxels.uv;
    return result;
}
