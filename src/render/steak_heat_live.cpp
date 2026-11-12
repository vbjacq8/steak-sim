// Steak heat live: the steak_heat_demo cook played on a clock. Same mesh, voxels, heat and cooked color,
// but the window opens on the raw steak at time zero and waits. Adjust the settings, press Start, and the
// view redraws as the steak cooks while a clock counts up to HeatParams::duration; the speed control runs
// that clock faster than real time. The voxel volume and the textured mesh switch with V, as in the demo.
// See showLiveHeatGrid in heat_view.hpp for the controls.
//
// Usage: steak_heat_live [mesh.obj|mesh.gltf|mesh.glb]

#include "demo_mesh.hpp"

#include <string>

int main(int argc, char** argv){
    return runLiveDemo("steak_heat_live", argc > 1 ? argv[1] : STEAK_OBJ, STEAK_TEXTURE, HeatParams{});
}
