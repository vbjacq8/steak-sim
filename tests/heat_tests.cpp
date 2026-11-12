#include "compute_context.hpp"
#include "heat_solver.hpp"
#include "steak_pipeline.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

namespace {

// Shared by all tests; nullptr if no usable Vulkan device, in which case the tests skip.
struct Gpu{
    ComputeContext ctx{"heat_tests"};
    HeatSolver solver{ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, FINITE_DIFFERENCE_SPV};
};

Gpu* gpu(){
    static std::unique_ptr<Gpu> instance = []() -> std::unique_ptr<Gpu>{
        try{
            return std::make_unique<Gpu>();
        } catch (const std::exception& e){
            std::cerr << "Vulkan unavailable: " << e.what() << std::endl;
            return nullptr;
        }
    }();
    return instance.get();
}

#define REQUIRE_VULKAN() \
    if (gpu() == nullptr) GTEST_SKIP() << "no usable Vulkan device"

// Same arithmetic, in the same order, as heatStepMain in finitedifference.slang.
std::vector<float> cpuStep(const HeatParams& params, const std::vector<uint32_t>& cells, const std::vector<float>& in){
    const glm::uvec3 n = params.dims;
    const float r = params.fourier();
    std::vector<float> out(in.size());
    for (uint32_t z = 0; z < n.z; ++z){
        for (uint32_t y = 0; y < n.y; ++y){
            for (uint32_t x = 0; x < n.x; ++x){
                const size_t i = x + size_t(n.x) * (y + size_t(n.y) * z);
                const float c = in[i];
                if (cells[i] == CELL_PAN){
                    out[i] = params.panTemp;
                    continue;
                }
                if (cells[i] == CELL_AIR){
                    out[i] = params.airTemp;
                    continue;
                }
                const size_t sy = n.x, sz = size_t(n.x) * n.y;
                auto flux = [&](size_t j){
                    const float difference = in[j] - c;
                    if (cells[j] == CELL_AIR) return params.airCoupling() * difference;
                    if (cells[j] == CELL_PAN) return params.panCoupling() * difference;
                    return difference;
                };
                const float sum = (flux(x > 0 ? i - 1 : i) + flux(x + 1 < n.x ? i + 1 : i)) +
                                  (flux(y > 0 ? i - sy : i) + flux(y + 1 < n.y ? i + sy : i)) +
                                  (flux(z > 0 ? i - sz : i) + flux(z + 1 < n.z ? i + sz : i));
                out[i] = c + r * sum;
            }
        }
    }
    return out;
}

// Axis-aligned box [lo, hi] as 12 triangles.
void appendBox(std::vector<glm::vec3>& positions, std::vector<uint32_t>& indices, glm::vec3 lo, glm::vec3 hi){
    const auto base = static_cast<uint32_t>(positions.size());
    for (int corner = 0; corner < 8; ++corner){
        positions.emplace_back((corner & 1) ? hi.x : lo.x, (corner & 2) ? hi.y : lo.y, (corner & 4) ? hi.z : lo.z);
    }
    static constexpr uint32_t faces[12][3] = {
        {0, 2, 1}, {1, 2, 3}, {4, 5, 6}, {5, 7, 6},  // -z, +z
        {0, 1, 4}, {1, 5, 4}, {2, 6, 3}, {3, 6, 7},  // -y, +y
        {0, 4, 2}, {2, 4, 6}, {1, 3, 5}, {3, 7, 5},  // -x, +x
    };
    for (const auto& face : faces){
        for (uint32_t v : face) indices.push_back(base + v);
    }
}

TEST(HeatSolver, BlockCellsHavePanAtZZero){
    const glm::uvec3 dims(3, 4, 5);
    const std::vector<uint32_t> cells = makeBlockCells(dims);
    for (uint32_t z = 0; z < dims.z; ++z){
        for (uint32_t y = 0; y < dims.y; ++y){
            for (uint32_t x = 0; x < dims.x; ++x){
                EXPECT_EQ(cells[x + dims.x * (y + dims.y * z)], z == 0 ? CELL_PAN : CELL_SOLID);
            }
        }
    }
}

TEST(HeatSolver, PanCellsFollowVoxelFlags){
    const glm::uvec3 dims(2, 1, 3);
    const std::vector<uint32_t> voxels{0, 3, 1, 0, 2, 0};
    EXPECT_EQ(makePanCells(dims, voxels), (std::vector<uint32_t>{CELL_PAN, CELL_PAN, CELL_SOLID, CELL_AIR, CELL_SOLID, CELL_AIR}));
    EXPECT_THROW(makePanCells(dims, std::vector<uint32_t>(5)), std::invalid_argument);
}

TEST(HeatSolver, PanCellsAlongY){
    const glm::uvec3 dims(2, 3, 2);
    const std::vector<uint32_t> cells = makePanCells(dims, std::vector<uint32_t>(12, 1u), 1);
    for (uint32_t z = 0; z < dims.z; ++z){
        for (uint32_t y = 0; y < dims.y; ++y){
            for (uint32_t x = 0; x < dims.x; ++x){
                EXPECT_EQ(cells[x + dims.x * (y + dims.y * z)], y == 0 ? CELL_PAN : CELL_SOLID);
            }
        }
    }
    EXPECT_THROW(makePanCells(dims, std::vector<uint32_t>(12, 1u), 3), std::invalid_argument);
}

TEST(HeatSolver, UniformFieldStaysUniform){
    REQUIRE_VULKAN();
    const HeatParams params{.dims = {16, 16, 16}, .initialTemp = 70.0f, .panTemp = 70.0f};
    const HeatGrid grid = gpu()->solver.run(params, 20);
    EXPECT_FLOAT_EQ(grid.time, 20.0f);
    for (float t : grid.temperature) ASSERT_FLOAT_EQ(t, 70.0f);
}

TEST(HeatSolver, ZeroStepsReturnsInitialField){
    REQUIRE_VULKAN();
    const HeatParams params{.dims = {5, 6, 7}};
    std::vector<float> initial(5 * 6 * 7);
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = static_cast<float>(i);
    EXPECT_EQ(gpu()->solver.run(params, makeBlockCells(params.dims), initial, 0).temperature, initial);
}

// Odd sizes that aren't multiples of the 32x4x2 workgroup exercise the bounds checks and every boundary;
// random cell kinds exercise the air and pan rules.
TEST(HeatSolver, MatchesCpuStencil){
    REQUIRE_VULKAN();
    const HeatParams params{.dims = {19, 13, 7}, .panTemp = 400.0f};
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> temp(40.0f, 400.0f);
    std::discrete_distribution<uint32_t> kind({2.0, 7.0, 1.0});  // air, solid, pan
    std::vector<float> field(size_t(19) * 13 * 7);
    std::vector<uint32_t> cells(field.size());
    for (float& t : field) t = temp(rng);
    for (uint32_t& c : cells) c = kind(rng);

    const uint32_t steps = 5;
    const HeatGrid grid = gpu()->solver.run(params, cells, field, steps);
    EXPECT_EQ(grid.cells, cells);
    for (uint32_t s = 0; s < steps; ++s) field = cpuStep(params, cells, field);
    for (size_t i = 0; i < field.size(); ++i) ASSERT_NEAR(grid.temperature[i], field[i], 1e-3f) << "cell " << i;
}

// A semi-infinite solid at T0 whose surface is suddenly exposed to a reservoir Tr through h (Carslaw & Jaeger):
//   (T - T0) / (Tr - T0) = erfc(u) - exp(h d / k + h^2 alpha t / k^2) erfc(u + h sqrt(alpha t) / k),  u = d / (2 sqrt(alpha t)),
// where d is depth below the reservoir face; cell z is centered at d = (z - 1/2) dx. The grid is 64 mm deep, far
// beyond the diffusion length, so the insulated far face is never felt.
double convectiveSurfaceSolution(const HeatParams& p, double transfer, double reservoir, double depth, double time){
    const double alpha = p.diffusivity, k = p.conductivity, h = transfer;
    const double u = depth / (2.0 * std::sqrt(alpha * time));
    const double beta = h * std::sqrt(alpha * time) / k;
    const double fraction = std::erfc(u) - std::exp(h * depth / k + beta * beta) * std::erfc(u + beta);
    return p.initialTemp + (reservoir - p.initialTemp) * fraction;
}

TEST(HeatSolver, MatchesConvectiveSurfaceSolution){
    REQUIRE_VULKAN();
    for (float h : {25.0f, 500.0f}){
        const HeatParams params{.dims = {4, 4, 64}, .panTransfer = h};
        const HeatGrid grid = gpu()->solver.run(params, 240);
        for (uint32_t z = 1; z < 30; ++z){
            const double exact = convectiveSurfaceSolution(params, h, params.panTemp, (z - 0.5) * params.spacing, grid.time);
            EXPECT_NEAR(grid.at(1, 2, z), exact, 0.01 * (params.panTemp - params.initialTemp)) << "h = " << h << ", z = " << z;
        }
    }
}

// Air convects at HeatParams::airTransfer. The same semi-infinite solution as the pan, with the z = 0 layer
// held at airTemp instead of panTemp. h = 10 and h = 30 bracket vapor around a cooking steak.
TEST(HeatSolver, AirConvectionMatchesSurfaceSolution){
    REQUIRE_VULKAN();
    for (float h : {10.0f, 30.0f}){
        const HeatParams params{.dims = {4, 4, 64}, .airTransfer = h, .airTemp = 212.0f};
        std::vector<uint32_t> cells(params.dims.x * params.dims.y * params.dims.z, CELL_SOLID);
        for (uint32_t y = 0; y < params.dims.y; ++y){
            for (uint32_t x = 0; x < params.dims.x; ++x) cells[x + params.dims.x * y] = CELL_AIR;
        }
        const HeatGrid grid = gpu()->solver.run(params, cells, 240);
        EXPECT_FLOAT_EQ(grid.at(1, 2, 0), params.airTemp);
        for (uint32_t z = 1; z < 30; ++z){
            const double exact = convectiveSurfaceSolution(params, h, params.airTemp, (z - 0.5) * params.spacing, grid.time);
            EXPECT_NEAR(grid.at(1, 2, z), exact, 0.01 * (params.airTemp - params.initialTemp)) << "h = " << h << ", z = " << z;
        }
    }
}

// A layer of air at z = 3 is held at airTemp, so it carries no heat from the pan into the meat beyond it.
TEST(HeatSolver, AirGapBlocksHeat){
    REQUIRE_VULKAN();
    const HeatParams params{.dims = {8, 8, 12}};
    std::vector<uint32_t> cells = makeBlockCells(params.dims);
    for (uint32_t y = 0; y < 8; ++y){
        for (uint32_t x = 0; x < 8; ++x) cells[x + 8 * (y + 8 * 3)] = CELL_AIR;
    }
    const HeatGrid grid = gpu()->solver.run(params, cells, 200);
    EXPECT_GT(grid.at(4, 4, 2), params.initialTemp + 20.0f);
    EXPECT_FLOAT_EQ(grid.at(4, 4, 3), params.airTemp);
    EXPECT_LT(grid.at(4, 4, 4), grid.at(4, 4, 2));
    EXPECT_NEAR(grid.at(4, 4, 11), params.initialTemp, 0.1f);
}

// Batched submits used by the progress bar must finish on the same field as one submit.
TEST(HeatSolver, ProgressBatchesMatchSingleSubmit){
    REQUIRE_VULKAN();
    const HeatParams params{.dims = {8, 8, 8}, .panTemp = 400.0f};
    std::vector<float> seen;
    const HeatGrid batched = gpu()->solver.run(params, 40, [&](float fraction){ seen.push_back(fraction); });
    const HeatGrid once = gpu()->solver.run(params, 40);
    ASSERT_FALSE(seen.empty());
    EXPECT_GT(seen.front(), 0.0f);
    EXPECT_FLOAT_EQ(seen.back(), 1.0f);
    for (size_t i = 1; i < seen.size(); ++i) EXPECT_GE(seen[i], seen[i - 1]);
    EXPECT_EQ(batched.temperature, once.temperature);
}

// A live view advances a session in uneven batches and reads it back in between; it must land on the same field.
TEST(HeatSession, BatchesWithReadsMatchOneRun){
    REQUIRE_VULKAN();
    const HeatParams params{.dims = {9, 7, 6}, .panTemp = 400.0f};
    const std::vector<uint32_t> cells = makeBlockCells(params.dims);
    HeatSession session(gpu()->solver, params, cells);
    EXPECT_EQ(session.read().temperature, gpu()->solver.run(params, cells, 0).temperature);
    session.advance(7);
    const HeatGrid partway = session.read();
    EXPECT_FLOAT_EQ(partway.time, 7.0f);
    EXPECT_EQ(partway.temperature, gpu()->solver.run(params, cells, 7).temperature);
    session.advance(0);
    session.advance(13);
    const HeatGrid grid = session.read();
    EXPECT_EQ(session.stepsDone(), 20u);
    EXPECT_FLOAT_EQ(grid.time, 20.0f);
    EXPECT_EQ(grid.temperature, gpu()->solver.run(params, cells, 20).temperature);
}

TEST(HeatSolver, RejectsUnstableTimeStep){
    REQUIRE_VULKAN();
    const HeatParams params{.dims = {8, 8, 8}, .spacing = 5.0e-4f, .timeStep = 1.0f};
    ASSERT_GT(params.fourier(), params.maxStableFourier());
    EXPECT_THROW(gpu()->solver.run(params, 1), std::invalid_argument);
}

// Default scenario: 128^3, dx = 1 mm, dt = 1 s, 60 s, HeatParams::panTransfer. Only z = 0 is heated and the other faces are
// insulated, so the field depends on z alone, stays within [initial, pan], and cools away from the pan.
TEST(HeatSolver, DefaultScenarioIsOneDimensionalAndBounded){
    REQUIRE_VULKAN();
    const HeatParams params;
    const HeatGrid grid = gpu()->solver.run(params, 60);
    ASSERT_EQ(grid.dims, glm::uvec3(128));

    for (uint32_t z = 0; z < grid.dims.z; ++z){
        const float column = grid.at(0, 0, z);
        if (z == 0) EXPECT_FLOAT_EQ(column, params.panTemp);
        if (z > 0) EXPECT_LE(column, grid.at(0, 0, z - 1)) << "z = " << z;
        EXPECT_GE(column, params.initialTemp);
        for (uint32_t y = 0; y < grid.dims.y; y += 9){
            for (uint32_t x = 0; x < grid.dims.x; x += 7) ASSERT_NEAR(grid.at(x, y, z), column, 1e-3f);
        }
    }
    EXPECT_GT(grid.at(0, 0, 1), params.initialTemp + 10.0f);
    EXPECT_LT(grid.at(0, 0, 1), params.panTemp);
    EXPECT_FLOAT_EQ(grid.at(0, 0, 64), params.initialTemp);
}

// A 2 x 2 x 1 cm box resting on z = 0. Heat comes in from the pan layer; cells are HeatParams::spacing wide.
TEST(SteakPipeline, VoxelizedBoxHeatsFromThePan){
    REQUIRE_VULKAN();
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    appendBox(positions, indices, {0.0f, 0.0f, 0.0f}, {2.0f, 2.0f, 1.0f});

    const HeatParams params;
    const SteakHeatResult result = simulateMeshHeat(gpu()->ctx, positions, indices, 0.01f, params, 60);
    const HeatGrid& heat = result.heat;
    ASSERT_EQ(heat.dims, result.voxels.dims);
    EXPECT_FLOAT_EQ(heat.spacing, params.spacing);
    EXPECT_NEAR(result.voxels.voxelSize, params.spacing / 0.01f, 1e-5f);
    EXPECT_FLOAT_EQ(heat.time, 60.0f);

    size_t solid = 0;
    for (size_t i = 0; i < heat.cellCount(); ++i){
        const bool panLayer = i / (size_t(heat.dims.x) * heat.dims.y) == 0;
        const uint32_t expected = panLayer ? CELL_PAN : (result.voxels.cells[i] != 0 ? CELL_SOLID : CELL_AIR);
        ASSERT_EQ(heat.cells[i], expected) << "cell " << i;
        if (heat.cells[i] == CELL_AIR) ASSERT_FLOAT_EQ(heat.temperature[i], params.airTemp);
        if (heat.cells[i] == CELL_PAN) ASSERT_FLOAT_EQ(heat.temperature[i], params.panTemp);
        solid += heat.cells[i] == CELL_SOLID;
    }
    EXPECT_GT(solid, 0u);

    // The box's low-z layer sits on the pan and warms; farther from the pan it warms less.
    const glm::uvec3 c = heat.dims / 2u;
    ASSERT_EQ(heat.kind(c.x, c.y, 1), CELL_SOLID);
    EXPECT_GT(heat.at(c.x, c.y, 1), params.initialTemp);
    const uint32_t far = std::min(heat.dims.z - 1, 4u);
    if (far > 1 && heat.kind(c.x, c.y, far) == CELL_SOLID){
        EXPECT_LE(heat.at(c.x, c.y, far), heat.at(c.x, c.y, 1));
    }
}

// The same kind of box standing on its side: with panAxis = 1 the y = 0 layer is the pan and heat comes in along y.
TEST(SteakPipeline, PanAlongYHeatsTheLowYFace){
    REQUIRE_VULKAN();
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    appendBox(positions, indices, {0.0f, 0.0f, 0.0f}, {2.0f, 1.0f, 2.0f});
    HeatParams params;
    params.panAxis = 1;
    const SteakHeatResult result = simulateMeshHeat(gpu()->ctx, positions, indices, 0.01f, params, 60);
    const HeatGrid& heat = result.heat;
    EXPECT_EQ(heat.panAxis, 1u);
    for (uint32_t z = 0; z < heat.dims.z; ++z){
        for (uint32_t x = 0; x < heat.dims.x; ++x) ASSERT_EQ(heat.kind(x, 0, z), CELL_PAN);
    }
    const glm::uvec3 c = heat.dims / 2u;
    ASSERT_EQ(heat.kind(c.x, 1, c.z), CELL_SOLID);
    ASSERT_NE(heat.kind(c.x, c.y, 0), CELL_PAN);
    EXPECT_GT(heat.at(c.x, 1, c.z), params.initialTemp + 10.0f);
    EXPECT_GT(heat.at(c.x, 1, c.z), heat.at(c.x, heat.dims.y - 2, c.z));
}

}  // namespace
