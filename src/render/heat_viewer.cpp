// Heat viewer: runs the finite-difference heat solver on a solid block whose y = 0 layer is the pan, then
// shows the result (see heat_view.hpp for the keys). The scenario is the HeatParams defaults.

#include "compute_context.hpp"
#include "heat_solver.hpp"
#include "heat_view.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>

int main(){
    const HeatParams params;
    const auto steps = static_cast<uint32_t>(std::lround(params.duration / params.timeStep));

    try{
        HeatGrid grid;
        {
            ComputeContext ctx("heat_viewer");
            HeatSolver solver(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, FINITE_DIFFERENCE_SPV);
            const auto start = std::chrono::steady_clock::now();
            grid = solver.run(params, steps);
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
            std::cout << "Ran " << steps << " steps (dt " << params.timeStep << " s, r = " << params.fourier() << ") on a "
                      << grid.dims.x << "x" << grid.dims.y << "x" << grid.dims.z << " grid in " << elapsed.count() << " ms\n";
        }
        printCenterColumn(grid, 16);
        showHeatGrid("heat viewer", std::move(grid), params, steps,
                     [](const HeatParams& next, uint32_t nextSteps, const std::function<void(float)>& progress){
                         ComputeContext ctx("heat_viewer");
                         HeatSolver solver(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, FINITE_DIFFERENCE_SPV);
                         return solver.run(next, nextSteps, progress);
                     });
    } catch (const std::exception& e){
        std::cerr << "heat_viewer: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
