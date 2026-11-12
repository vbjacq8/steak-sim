// Finds the surface temperature that goes with a measured center temperature, for calibrating the
// doneness table in src/sim/color_model.hpp against photos of steaks cooked to a known center.
//
// The steak is a single column of cells: pan, `thickness` of meat, air. Grid edges are insulated, so
// the column is a 1D slab with the heat solver's defaults (HeatParams). It is cooked one side, flipped,
// and cooked the same time on the other side. The per-side time is searched until the hottest
// center temperature reached equals the target. Each layer reports the hottest temperature it reached;
// the first layer on each face is what a crust voxel sees in the steak simulation.

#include "color_model.hpp"
#include "compute_context.hpp"
#include "heat_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Cook{
    std::vector<float> peak;  // degrees Fahrenheit, per meat layer, 0 = the side that went down first
    float center = 0.0f;      // hottest center temperature, F
};

class Column{
public:
    Column(const ComputeContext& ctx, HeatParams paramsIn, uint32_t layersIn)
        : solver(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, FINITE_DIFFERENCE_SPV),
          params(paramsIn),
          layers(layersIn){
        params.dims = {1, 1, layers + 2};
        cells.assign(layers + 2, CELL_SOLID);
        cells.front() = CELL_PAN;
        cells.back() = CELL_AIR;
    }

    Cook cook(uint32_t stepsPerSide){
        std::vector<float> field(layers + 2, params.initialTemp);
        field.front() = params.panTemp;
        field.back() = params.airTemp;
        Cook result;
        result.peak.assign(layers, params.initialTemp);
        result.center = params.initialTemp;
        for (int side = 0; side < 2; ++side){
            for (uint32_t step = 0; step < stepsPerSide; ++step){
                field = solver.run(params, cells, field, 1).temperature;
                for (uint32_t k = 0; k < layers; ++k){
                    const uint32_t original = side == 0 ? k : layers - 1 - k;
                    result.peak[original] = std::max(result.peak[original], field[k + 1]);
                }
                result.center = std::max(result.center, centerOf(field));
            }
            std::reverse(field.begin() + 1, field.end() - 1);
        }
        return result;
    }

    float centerOf(const std::vector<float>& field) const{
        if (layers % 2 == 1) return field[1 + layers / 2];
        return 0.5f * (field[layers / 2] + field[1 + layers / 2]);
    }

    const HeatParams& heatParams() const { return params; }

private:
    HeatSolver solver;
    HeatParams params;
    uint32_t layers;
    std::vector<uint32_t> cells;
};

float toCelsius(float fahrenheit){ return fahrenheitToCelsius(fahrenheit); }

void printCook(const char* title, const Cook& cook, uint32_t stepsPerSide, float timeStep, float spacingMm){
    std::printf("%s: %.1f min per side (%u s), hottest center %.1f F (%.1f C)\n", title, stepsPerSide * timeStep / 60.0f,
                stepsPerSide, cook.center, toCelsius(cook.center));
    const size_t layers = cook.peak.size();
    for (size_t k = 0; k < layers; ++k){
        const char* note = k == 0 ? "  <- surface, down first" : (k + 1 == layers ? "  <- surface, down second" : "");
        std::printf("  %5.1f mm  peak %6.1f F  %6.1f C%s\n", (static_cast<float>(k) + 0.5f) * spacingMm, cook.peak[k],
                    toCelsius(cook.peak[k]), note);
    }
    const float surface = 0.5f * (cook.peak.front() + cook.peak.back());
    std::printf("  characteristic surface (mean of both faces): %.1f F (%.1f C)\n", surface, toCelsius(surface));
}

int usage(){
    std::fprintf(stderr,
                 "usage: doneness_calibrate <center F> [--thickness-mm 25.4] [--per-side-min M]\n"
                 "  Finds the per-side time (one flip) that brings the center to <center F> and prints the\n"
                 "  hottest temperature of every layer. --per-side-min also cooks for exactly that time.\n");
    return EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv){
    if (argc < 2) return usage();
    float targetF = 0.0f;
    float thicknessMm = 25.4f;
    float checkMinutes = -1.0f;
    try{
        targetF = std::stof(argv[1]);
        for (int i = 2; i < argc; ++i){
            const std::string flag = argv[i];
            if (i + 1 >= argc) return usage();
            if (flag == "--thickness-mm") thicknessMm = std::stof(argv[++i]);
            else if (flag == "--per-side-min") checkMinutes = std::stof(argv[++i]);
            else return usage();
        }
    } catch (const std::exception&){
        return usage();
    }

    try{
        const HeatParams defaults;
        const auto layers = static_cast<uint32_t>(std::max(1L, std::lround(thicknessMm / (defaults.spacing * 1000.0f))));
        const float spacingMm = defaults.spacing * 1000.0f;
        std::printf("Column: %u layers of %.1f mm (%.1f mm), pan %.0f F, air %.0f F, start %.0f F, dt %.2f s\n", layers, spacingMm,
                    layers * spacingMm, defaults.panTemp, defaults.airTemp, defaults.initialTemp, defaults.timeStep);
        if (!(targetF > defaults.initialTemp)) throw std::invalid_argument("center target must be above the starting temperature");

        ComputeContext ctx("doneness_calibrate");
        Column column(ctx, defaults, layers);
        const float dt = defaults.timeStep;

        // The hottest center rises with time per side; find the shortest time that reaches the target.
        uint32_t lo = 1;
        uint32_t hi = static_cast<uint32_t>(std::lround(60.0f / dt));
        while (column.cook(hi).center < targetF){
            lo = hi;
            hi *= 2;
            if (hi * dt > 3600.0f) throw std::runtime_error("center never reaches the target within an hour per side");
        }
        while (hi - lo > 1){
            const uint32_t mid = lo + (hi - lo) / 2;
            (column.cook(mid).center < targetF ? lo : hi) = mid;
        }
        printCook("Matched", column.cook(hi), hi, dt, spacingMm);

        if (checkMinutes > 0.0f){
            const auto steps = static_cast<uint32_t>(std::max(1L, std::lround(checkMinutes * 60.0f / dt)));
            printCook("Photo time", column.cook(steps), steps, dt, spacingMm);
        }
    } catch (const std::exception& e){
        std::fprintf(stderr, "doneness_calibrate: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
