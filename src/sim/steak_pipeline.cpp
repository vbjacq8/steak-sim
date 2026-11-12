#include "steak_pipeline.hpp"

#include <stdexcept>

namespace {

void copyColor(HeatGrid& heat, const ColorSolver& color, const VoxelGrid& voxels){
    heat.color = color.linearRgb();
    heat.initialColor = color.initialLinearRgb();
    heat.peakCelsius = color.peakCelsius();
    heat.colorReferenceY = color.reference().luminance;
    heat.colorReferenceA = color.reference().redness;
    heat.sampleUv = voxels.uv;
}

VoxelGrid voxelizeTextured(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                           const std::vector<glm::vec2>& texcoords, const std::vector<uint32_t>& indices,
                           float metersPerUnit, float spacing){
    if (!(metersPerUnit > 0.0f)) throw std::invalid_argument("metersPerUnit must be positive");
    Voxelizer voxelizer(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, VOXELIZE_SPV);
    return voxelizer.voxelize(positions, texcoords, indices, spacing / metersPerUnit);
}

}  // namespace

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
    const std::vector<uint32_t> cells = makeCookCells(params, result.voxels.cells);
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
    result.voxels = voxelizeTextured(ctx, positions, texcoords, indices, metersPerUnit, params.spacing);
    if (progress) progress(0.2f);

    params.dims = result.voxels.dims;
    const std::vector<uint32_t> cells = makeCookCells(params, result.voxels.cells);
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
    copyColor(result.heat, color, result.voxels);
    return result;
}

MeshHeatColorSession::MeshHeatColorSession(const ComputeContext& ctx, const std::vector<glm::vec3>& positions,
                                           const std::vector<glm::vec2>& texcoords, const std::vector<uint32_t>& indices,
                                           float metersPerUnit, HeatParams params, const uint8_t* rgba,
                                           uint32_t textureWidth, uint32_t textureHeight)
    : voxelGrid(voxelizeTextured(ctx, positions, texcoords, indices, metersPerUnit, params.spacing)),
      color(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, COLOR_INIT_SPV, COLOR_KINETICS_SPV, COLOR_EXPORT_SPV),
      solver(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, FINITE_DIFFERENCE_SPV){
    params.dims = voxelGrid.dims;
    const std::vector<uint32_t> cells = makeCookCells(params, voxelGrid.cells);
    color.prepareGrid(voxelGrid.dims, cells, voxelGrid.uv, rgba, textureWidth, textureHeight, params.timeStep);
    session.emplace(solver, params, cells, &color);
}

HeatGrid MeshHeatColorSession::read(){
    HeatGrid heat = session->read();
    color.download();
    copyColor(heat, color, voxelGrid);
    return heat;
}
