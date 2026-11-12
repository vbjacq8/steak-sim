// Steak heat live: the steak_heat_demo cook played on a clock. Same mesh, voxels, heat and cooked color,
// but the window opens on the raw steak at time zero and waits. Adjust the settings, press Start, and the
// view redraws as the steak cooks while a clock counts up to HeatParams::duration; the speed control runs
// that clock faster than real time. The voxel volume and the textured mesh switch with V, as in the demo.
// See showLiveHeatGrid in heat_view.hpp for the controls.
//
// Usage: steak_heat_live [mesh.obj|mesh.gltf|mesh.glb]

#include "demo_mesh.hpp"
#include "heat_view.hpp"
#include "steak_pipeline.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

class SteakLive : public LiveHeat{
public:
    explicit SteakLive(DemoMesh meshIn) : mesh(std::move(meshIn)){}

    HeatGrid restart(const HeatParams& params) override{
        checkCellCount(mesh, params.spacing);
        run.reset();
        run = std::make_unique<MeshHeatColorSession>(ctx, mesh.positions, mesh.solverUvs, mesh.indices, mesh.metersPerUnit, params,
                                                     mesh.atlas.pixels.data(), static_cast<uint32_t>(mesh.atlas.width),
                                                     static_cast<uint32_t>(mesh.atlas.height));
        const glm::uvec3 dims = run->voxels().dims;
        std::cout << "Voxelized into a " << dims.x << "x" << dims.y << "x" << dims.z << " grid (dx "
                  << params.spacing * 1000.0f << " mm, dt " << params.timeStep << " s, r = " << params.fourier() << ")\n";
        return run->read();
    }

    HeatGrid advance(uint32_t steps) override{
        if (!run) throw std::logic_error("steak_heat_live: advance before restart");
        run->advance(steps);
        return run->read();
    }

private:
    DemoMesh mesh;
    ComputeContext ctx{"steak_heat_live"};
    std::unique_ptr<MeshHeatColorSession> run;
};

}  // namespace

int main(int argc, char** argv){
    const std::string objPath = argc > 1 ? argv[1] : STEAK_OBJ;
    try{
        DemoMesh mesh = prepareDemoMesh(loadMesh(objPath), STEAK_TEXTURE);
        std::cout << "Mesh scale: 1 unit = " << mesh.metersPerUnit * 100.0f << " cm, long axis "
                  << longestAxis(mesh.positions) * mesh.metersPerUnit * 100.0f << " cm, pan on the low "
                  << "xyz"[mesh.panAxis] << " side\n";
        HeatParams params;
        params.panAxis = mesh.panAxis;
        HeatSurface surface = demoSurface(mesh);
        showLiveHeatGrid("steak heat live: " + objPath, params, std::make_unique<SteakLive>(std::move(mesh)), true,
                         std::move(surface));
    } catch (const std::exception& e){
        std::cerr << "steak_heat_live: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
