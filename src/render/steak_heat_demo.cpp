// Steak heat demo: steak mesh -> GPU voxelizer -> GPU heat solver -> cooked color -> block view.
//
// The mesh is in centimeters and is voxelized with cells HeatParams::spacing wide. Every voxel the steak
// touches conducts heat. The empty layer against the model's thinnest side (y for steak1, z for steak2)
// is the pan, and the air around it convects at HeatParams::airTransfer. After each heat step, every
// voxel's color follows the doneness table for the
// hottest temperature it has reached. The window starts on the temperature field; C switches solid
// cells to that cooked color so the two can be compared. The pan stays on the temperature key.
// V switches from the voxel blocks to the textured mesh. Cooked color on that mesh samples the voxels.
// Interior voxels take the UV of the nearest surface point, so the crust texture bleeds inward.
// See heat_view.hpp for the viewer keys.
//
// Usage: steak_heat_demo [mesh.obj|mesh.gltf|mesh.glb]
//
// The model is prepared as described in demo_mesh.hpp. steak_heat_live plays the same cook on a clock.

#include "demo_mesh.hpp"
#include "heat_view.hpp"
#include "steak_pipeline.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv){
    const std::string objPath = argc > 1 ? argv[1] : STEAK_OBJ;
    HeatParams params;
    const auto steps = static_cast<uint32_t>(std::lround(params.duration / params.timeStep));

    try{
        const Mesh loaded = loadMesh(objPath);
        const DemoMesh mesh = prepareDemoMesh(loaded, STEAK_TEXTURE);
        params.panAxis = mesh.panAxis;
        std::cout << "Mesh scale: 1 unit = " << mesh.metersPerUnit * 100.0f << " cm, long axis "
                  << longestAxis(mesh.positions) * mesh.metersPerUnit * 100.0f << " cm, pan on the low "
                  << "xyz"[mesh.panAxis] << " side\n";
        SteakHeatResult result;
        {
            ComputeContext ctx("steak_heat_demo");
            const auto start = std::chrono::steady_clock::now();
            result = simulateMeshHeatColor(ctx, mesh.positions, mesh.solverUvs, mesh.indices, mesh.metersPerUnit, params, steps,
                                           mesh.atlas.pixels.data(), static_cast<uint32_t>(mesh.atlas.width),
                                           static_cast<uint32_t>(mesh.atlas.height));
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
            const HeatGrid& heat = result.heat;
            std::cout << "Voxelized " << mesh.indices.size() / 3 << " triangles into a " << heat.dims.x << "x" << heat.dims.y
                      << "x" << heat.dims.z << " grid (dx " << heat.spacing * 1000.0f << " mm), then ran " << steps
                      << " heat and color steps (dt " << params.timeStep << " s, r = " << params.fourier() << ") in "
                      << elapsed.count() << " ms\n";
        }
        printCenterColumn(result.heat, result.heat.dims[result.heat.panAxis]);
        HeatSurface surface = demoSurface(mesh);
        showHeatGrid(
            "steak heat demo: " + objPath, std::move(result.heat), params, steps,
            [mesh](const HeatParams& next, uint32_t nextSteps, const std::function<void(float)>& progress){
                checkCellCount(mesh, next.spacing);
                ComputeContext ctx("steak_heat_demo");
                return simulateMeshHeatColor(ctx, mesh.positions, mesh.solverUvs, mesh.indices, mesh.metersPerUnit, next, nextSteps,
                                             mesh.atlas.pixels.data(), static_cast<uint32_t>(mesh.atlas.width),
                                             static_cast<uint32_t>(mesh.atlas.height), progress)
                    .heat;
            },
            true, std::move(surface));
    } catch (const std::exception& e){
        std::cerr << "steak_heat_demo: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
