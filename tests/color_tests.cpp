#include "color_model.hpp"
#include "color_solver.hpp"
#include "compute_context.hpp"
#include "steak_pipeline.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {

struct Gpu{
    ComputeContext ctx{"color_tests"};
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

float channelError(glm::vec3 a, glm::vec3 b){
    return std::max(std::abs(a.x - b.x), std::max(std::abs(a.y - b.y), std::abs(a.z - b.z)));
}

float simulateRamp(float dt, float totalSeconds){
    const int steps = static_cast<int>(std::lround(totalSeconds / dt));
    float a = 25.0f;
    for (int i = 0; i < steps; ++i){
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(steps);
        const float celsius = 30.0f + u * 60.0f;
        const float fahrenheit = celsius * 9.0f / 5.0f + 32.0f;
        a = stepLabA(a, 25.0f, fahrenheit, dt);
    }
    return a;
}

TEST(ColorModel, LabRoundTrip){
    const glm::vec3 white = linearSrgbToLab({1.0f, 1.0f, 1.0f});
    EXPECT_NEAR(white.x, 100.0f, 1e-3f);
    EXPECT_NEAR(white.y, 0.0f, 1e-3f);
    EXPECT_NEAR(white.z, 0.0f, 1e-3f);

    const glm::vec3 samples[] = {{0, 0, 0}, {1, 1, 1}, {0.5f, 0.5f, 0.5f}, {1, 0, 0}, {0, 1, 0},
                                 {0, 0, 1}, {0.6f, 0.25f, 0.2f}, {0.001f, 0.001f, 0.001f}};
    for (glm::vec3 linear : samples){
        EXPECT_LT(channelError(labToLinearSrgb(linearSrgbToLab(linear)), linear), 2e-3f);
        const glm::vec3 encoded = linearToSrgb(linear);
        const glm::vec3 back = linearToSrgb(labToLinearSrgb(linearSrgbToLab(srgbToLinear(encoded))));
        EXPECT_LT(channelError(back, encoded), 2e-3f);
    }
}

TEST(ColorModel, KineticsMovesTowardEquilibrium){
    const float cold = arrheniusK(fahrenheitToCelsius(100.0f) + 273.15f);
    const float hot = arrheniusK(fahrenheitToCelsius(300.0f) + 273.15f);
    EXPECT_GT(hot, cold);
    EXPECT_LT(equilibriumA(20.0f, 90.0f), equilibriumA(20.0f, 40.0f));
    EXPECT_FLOAT_EQ(stepLabA(12.0f, 20.0f, -500.0f, 1.0f), 12.0f);

    const float a0 = 20.0f;
    const float aInf = equilibriumA(a0, fahrenheitToCelsius(212.0f));
    float a = a0;
    float previous = std::abs(a - aInf);
    for (int i = 0; i < 80; ++i){
        a = stepLabA(a, a0, 212.0f, 1.0f);
        const float distance = std::abs(a - aInf);
        EXPECT_LT(distance, previous);
        previous = distance;
    }
    EXPECT_LT(std::abs(a - aInf), 0.2f);
}

TEST(ColorModel, SmallerTimestepApproachesFineReference){
    const float fine = simulateRamp(0.01f, 60.0f);
    const float mid = simulateRamp(0.1f, 60.0f);
    const float coarse = simulateRamp(1.0f, 60.0f);
    EXPECT_GT(std::abs(coarse - fine), 1e-3f);
    EXPECT_LT(std::abs(mid - fine), std::abs(coarse - fine));
}

std::vector<uint8_t> texel(uint8_t r, uint8_t g, uint8_t b){
    return {r, g, b, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
}

glm::vec3 cpuLabOfTexel(int x, int y){
    const std::vector<uint8_t> image = texel(255, 0, 0);
    const int i = (y * 2 + x) * 4;
    return linearSrgbToLab(srgbToLinear({image[i] / 255.0f, image[i + 1] / 255.0f, image[i + 2] / 255.0f}));
}

ColorSolver makeSolver(){
    return ColorSolver(gpu()->ctx.device, gpu()->ctx.physicalDevice, gpu()->ctx.queue, gpu()->ctx.queueFamily, COLOR_INIT_SPV,
                       COLOR_KINETICS_SPV, COLOR_EXPORT_SPV);
}

TEST(ColorSolver, InitMatchesTextureLab){
    REQUIRE_VULKAN();
    const std::vector<uint8_t> image = texel(255, 0, 0);
    const glm::uvec3 dims{5, 1, 1};
    std::vector<uint32_t> cells(5, CELL_SOLID);
    cells[4] = CELL_AIR;
    const std::vector<glm::vec2> uv{{0.25f, 0.25f}, {0.75f, 0.25f}, {0.25f, 0.75f}, {0.75f, 0.75f}, {0.0f, 0.0f}};
    const glm::ivec2 pixel[4] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};

    ColorSolver solver = makeSolver();
    solver.prepareGrid(dims, cells, uv, image.data(), 2, 2, 1.0f);
    solver.runExplicitTemperatures({});

    ASSERT_EQ(solver.lab().size(), 5u);
    for (int i = 0; i < 4; ++i){
        const glm::vec3 expected = cpuLabOfTexel(pixel[i].x, pixel[i].y);
        EXPECT_NEAR(solver.lab()[i].x, expected.x, 0.15f) << i;
        EXPECT_NEAR(solver.lab()[i].y, expected.y, 0.15f) << i;
        EXPECT_NEAR(solver.lab()[i].z, expected.z, 0.15f) << i;
        EXPECT_NEAR(solver.lab()[i].w, 1.0f, 1e-3f);
        const glm::vec3 rgb = labToLinearSrgb(glm::vec3(solver.lab()[i]));
        EXPECT_NEAR(solver.linearRgb()[i].x, rgb.x, 0.02f);
        EXPECT_NEAR(solver.linearRgb()[i].y, rgb.y, 0.02f);
        EXPECT_NEAR(solver.linearRgb()[i].z, rgb.z, 0.02f);
    }
    EXPECT_NEAR(solver.lab()[4].w, 0.0f, 1e-3f);
    EXPECT_NEAR(solver.linearRgb()[4].x, 0.0f, 1e-3f);
}

TEST(ColorSolver, GpuKineticsMatchesCpu){
    REQUIRE_VULKAN();
    const std::vector<uint8_t> image = texel(255, 0, 0);
    const glm::uvec3 dims{2, 1, 1};
    const std::vector<uint32_t> cells{CELL_SOLID, CELL_AIR};
    const std::vector<glm::vec2> uv{{0.25f, 0.25f}, {0.25f, 0.25f}};
    const std::vector<float> schedule{70.0f, 120.0f, 180.0f, 250.0f, 320.0f};
    constexpr float dt = 2.0f;

    ColorSolver initial = makeSolver();
    initial.prepareGrid(dims, cells, uv, image.data(), 2, 2, dt);
    initial.runExplicitTemperatures({});
    const glm::vec4 lab0 = initial.lab()[0];

    std::vector<std::vector<float>> fields;
    for (float temperature : schedule) fields.push_back({temperature, temperature});
    ColorSolver solved = makeSolver();
    solved.prepareGrid(dims, cells, uv, image.data(), 2, 2, dt);
    solved.runExplicitTemperatures(fields);

    float a = lab0.y;
    for (float temperature : schedule) a = stepLabA(a, lab0.y, temperature, dt);
    EXPECT_NEAR(solved.lab()[0].x, lab0.x, 0.08f);
    EXPECT_NEAR(solved.lab()[0].z, lab0.z, 0.08f);
    EXPECT_NEAR(solved.lab()[0].y, a, 0.25f);
    EXPECT_LT(solved.lab()[0].y, lab0.y - 0.5f);
    EXPECT_NEAR(solved.lab()[1].w, 0.0f, 1e-3f);
    EXPECT_NEAR(glm::length(solved.linearRgb()[1]), 0.0f, 1e-3f);
}

void appendBox(std::vector<glm::vec3>& positions, std::vector<uint32_t>& indices, glm::vec3 lo, glm::vec3 hi){
    const auto base = static_cast<uint32_t>(positions.size());
    for (int corner = 0; corner < 8; ++corner){
        positions.emplace_back((corner & 1) ? hi.x : lo.x, (corner & 2) ? hi.y : lo.y, (corner & 4) ? hi.z : lo.z);
    }
    static constexpr uint32_t faces[12][3] = {
        {0, 2, 1}, {1, 2, 3}, {4, 5, 6}, {5, 7, 6}, {0, 1, 4}, {1, 5, 4},
        {2, 6, 3}, {3, 6, 7}, {0, 4, 2}, {2, 4, 6}, {1, 3, 5}, {3, 7, 5},
    };
    for (const auto& face : faces){
        indices.push_back(base + face[0]);
        indices.push_back(base + face[1]);
        indices.push_back(base + face[2]);
    }
}

TEST(ColorSolver, HeatIsUnchangedAndMeatGetsColor){
    REQUIRE_VULKAN();
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    appendBox(positions, indices, {0.0f, 0.0f, 0.0f}, {2.0f, 1.0f, 2.0f});
    const std::vector<glm::vec2> uvs(positions.size(), glm::vec2(0.25f, 0.25f));
    const std::vector<uint8_t> image = texel(180, 40, 30);

    HeatParams params;
    params.initialTemp = 180.0f;
    constexpr uint32_t steps = 15;
    const SteakHeatResult plain = simulateMeshHeat(gpu()->ctx, positions, indices, 0.01f, params, steps);
    const SteakHeatResult colored = simulateMeshHeatColor(gpu()->ctx, positions, uvs, indices, 0.01f, params, steps,
                                                         image.data(), 2, 2);

    ASSERT_EQ(colored.heat.dims, plain.heat.dims);
    ASSERT_EQ(colored.heat.cells, plain.heat.cells);
    ASSERT_EQ(colored.heat.temperature.size(), plain.heat.temperature.size());
    ASSERT_EQ(colored.heat.color.size(), colored.heat.cellCount());
    for (size_t i = 0; i < plain.heat.temperature.size(); ++i){
        EXPECT_FLOAT_EQ(colored.heat.temperature[i], plain.heat.temperature[i]);
        if (plain.heat.cells[i] == CELL_SOLID){
            EXPECT_GT(glm::length(colored.heat.color[i]), 0.05f);
        } else{
            EXPECT_NEAR(glm::length(colored.heat.color[i]), 0.0f, 1e-4f);
        }
    }
}

}  // namespace
