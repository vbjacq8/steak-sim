// Steak heat demo: steak mesh -> GPU voxelizer -> GPU heat solver -> heat-map view.
//
// The mesh is in centimeters and is voxelized with cells HeatParams::spacing wide. Every voxel the steak
// touches conducts heat. The empty layer under the steak is the pan, and the air around it convects at
// HeatParams::airTransfer. The scenario is the HeatParams defaults. See heat_view.hpp for the viewer keys.
//
// Usage: steak_heat_demo [model.obj]

#include "heat_view.hpp"
#include "steak_pipeline.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Mesh{
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
};

Mesh loadObj(const std::string& path){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) throw std::runtime_error(warn + err);

    Mesh mesh;
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3){
        mesh.positions.push_back({attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]});
    }
    for (const auto& shape : shapes){
        for (const auto& index : shape.mesh.indices) mesh.indices.push_back(static_cast<uint32_t>(index.vertex_index));
    }
    return mesh;
}

constexpr float METERS_PER_MODEL_UNIT = 0.01f;  // steak.obj is in centimeters

}  // namespace

int main(int argc, char** argv){
    const std::string objPath = argc > 1 ? argv[1] : STEAK_OBJ;
    const HeatParams params;
    const auto steps = static_cast<uint32_t>(std::lround(params.duration / params.timeStep));

    try{
        const Mesh mesh = loadObj(objPath);
        SteakHeatResult result;
        {
            ComputeContext ctx("steak_heat_demo");
            const auto start = std::chrono::steady_clock::now();
            result = simulateMeshHeat(ctx, mesh.positions, mesh.indices, METERS_PER_MODEL_UNIT, params, steps);
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
            const HeatGrid& heat = result.heat;
            std::cout << "Voxelized " << mesh.indices.size() / 3 << " triangles into a " << heat.dims.x << "x" << heat.dims.y
                      << "x" << heat.dims.z << " grid (dx " << heat.spacing * 1000.0f << " mm), then ran " << steps
                      << " steps (dt " << params.timeStep << " s, r = " << params.fourier() << ") in " << elapsed.count()
                      << " ms\n";
        }
        printCenterColumn(result.heat, result.heat.dims.y);
        const std::vector<glm::vec3> positions = mesh.positions;
        const std::vector<uint32_t> indices = mesh.indices;
        showHeatGrid(
            "steak heat demo: " + objPath, std::move(result.heat), params, steps,
            [positions, indices](const HeatParams& next, uint32_t nextSteps, const std::function<void(float)>& progress){
                glm::vec3 low(std::numeric_limits<float>::infinity());
                glm::vec3 high(-std::numeric_limits<float>::infinity());
                for (const glm::vec3& point : positions){
                    low = glm::min(low, point);
                    high = glm::max(high, point);
                }
                const float voxelSize = next.spacing / METERS_PER_MODEL_UNIT;
                const float extent = std::max({high.x - low.x, high.y - low.y, high.z - low.z});
                if (!(voxelSize > 0.0f) || extent / voxelSize > 500.0f){
                    throw std::invalid_argument("Cell size is too small; the grid would exceed 512 cells on a side.");
                }
                ComputeContext ctx("steak_heat_demo");
                return simulateMeshHeat(ctx, positions, indices, METERS_PER_MODEL_UNIT, next, nextSteps, progress).heat;
            },
            true);
    } catch (const std::exception& e){
        std::cerr << "steak_heat_demo: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
