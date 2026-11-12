// Steak grill live: steak_heat_live on a grill instead of a pan. The boundary layer under the steak (y = 0
// for steak1, z = 0 for steak2: the model's thinnest axis) holds 5 bars at the pan temperature, spaced
// evenly across that layer and running across the steak; between the bars the steak sees air. So only
// the stripes over the bars sear, and the gaps cook slowly from the steam. Show pan (P) draws the bars.
// The bar count and width are under Pan in the settings and take effect on the next Start or Reset.
//
// Usage: steak_grill_live [mesh.obj|mesh.gltf|mesh.glb]

#include "demo_mesh.hpp"

#include <string>

int main(int argc, char** argv){
    HeatParams params;
    params.grillSpokes = 5;
    return runLiveDemo("steak_grill_live", argc > 1 ? argv[1] : STEAK_OBJ, STEAK_TEXTURE, params);
}
