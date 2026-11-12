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

CookState simulateRamp(float dt, float totalSeconds){
    const int steps = static_cast<int>(std::lround(totalSeconds / dt));
    CookState state;
    for (int i = 0; i < steps; ++i){
        const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(steps);
        const float celsius = 30.0f + u * 60.0f;
        const float fahrenheit = celsius * 9.0f / 5.0f + 32.0f;
        state = stepCook(state, fahrenheit, dt);
    }
    return state;
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

TEST(ColorModel, KineticsTableInterpolation){
    const KineticsParams cold = rednessParams(44.9f);
    EXPECT_EQ(cold.k1, 0.0f);
    EXPECT_EQ(cold.k2, 0.0f);
    EXPECT_EQ(rednessParams(std::nanf("")).k2, 0.0f);
    EXPECT_EQ(lightnessParams(29.9f).k1, 0.0f);
    EXPECT_FLOAT_EQ(lightnessParams(35.0f).k1, 0.5f * (0.0014f + 0.0027f));

    const KineticsParams at60 = rednessParams(60.0f);
    EXPECT_FLOAT_EQ(at60.n, 0.7669f);
    EXPECT_FLOAT_EQ(at60.m, 3.8473f);
    EXPECT_FLOAT_EQ(at60.k1, 0.0037f);
    EXPECT_FLOAT_EQ(at60.k2, 0.0328f);

    const KineticsParams at65 = rednessParams(65.0f);
    EXPECT_FLOAT_EQ(at65.n, 0.5f * (0.7669f + 0.7820f));
    EXPECT_FLOAT_EQ(at65.k2, 0.5f * (0.0328f + 0.0451f));

    const KineticsParams at90 = rednessParams(90.0f);
    const KineticsParams hot = rednessParams(400.0f);
    EXPECT_FLOAT_EQ(at90.n, 0.7866f);
    EXPECT_FLOAT_EQ(hot.n, at90.n);
    EXPECT_FLOAT_EQ(hot.m, at90.m);
    EXPECT_FLOAT_EQ(hot.k1, at90.k1);
    EXPECT_FLOAT_EQ(hot.k2, at90.k2);
    EXPECT_FLOAT_EQ(lightnessParams(400.0f).n, 1.4172f);
}

TEST(ColorModel, NothingChangesBelowTheTables){
    CookState below;
    for (int i = 0; i < 600; ++i) below = stepCook(below, 85.0f, 1.0f);
    EXPECT_EQ(below.lightness.ratio, 1.0f);
    EXPECT_EQ(below.lightness.reactant, 1.0f);
    EXPECT_EQ(below.redness.ratio, 1.0f);
    EXPECT_EQ(below.redness.reactant, 1.0f);
}

TEST(ColorModel, LightnessRisesAndLevelsOff){
    // Up to 55 C there is no second reaction, so L* rises to 1 + m and stays.
    EXPECT_NEAR(lightnessRatioAt(50.0f, 1800.0f), 1.0f + 0.3095f, 1e-3f);
    EXPECT_LT(lightnessRatioAt(50.0f, 60.0f), 1.1f);

    // At 70 C L* overshoots within a minute, then settles at n.
    float peak = 1.0f;
    for (float seconds = 1.0f; seconds <= 120.0f; seconds += 1.0f) peak = std::max(peak, lightnessRatioAt(70.0f, seconds));
    EXPECT_GT(peak, 1.45f);
    EXPECT_NEAR(lightnessRatioAt(70.0f, 1800.0f), 1.4320f, 0.01f);

    const glm::vec3 raw{45.0f, 13.55f, 12.92f};
    CookState done;
    for (int i = 0; i < 600; ++i) done = stepCook(done, 176.0f, 1.0f);
    const glm::vec3 cooked = cookedLab(raw, done);
    EXPECT_GT(cooked.x, 60.0f);
    EXPECT_LT(cooked.y, raw.y);
    EXPECT_EQ(cooked.z, raw.z);
    EXPECT_EQ(cookedLab({90.0f, 0.0f, 0.0f}, done).x, 100.0f);
}

TEST(ColorModel, RednessRisesThenFallsTowardEquilibrium){
    // At 80 C a* briefly rises above raw, then drops toward n.
    float peak = 1.0f;
    for (float seconds = 1.0f; seconds <= 120.0f; seconds += 1.0f) peak = std::max(peak, rednessRatioAt(80.0f, seconds));
    EXPECT_GT(peak, 1.03f);
    EXPECT_LT(peak, 1.2f);
    EXPECT_NEAR(rednessRatioAt(80.0f, 1800.0f), rednessParams(80.0f).n, 0.02f);
    EXPECT_LT(rednessRatioAt(80.0f, 600.0f), 0.85f);
}

TEST(ColorModel, ConstantTemperatureStepsMatchClosedForm){
    for (float celsius : {35.0f, 45.0f, 52.0f, 60.0f, 75.0f, 90.0f}){
        KineticsState lightness;
        KineticsState redness;
        for (int i = 0; i < 300; ++i){
            lightness = stepKinetics(lightness, lightnessParams(celsius), 2.0f);
            redness = stepKinetics(redness, rednessParams(celsius), 2.0f);
        }
        EXPECT_NEAR(lightness.ratio, lightnessRatioAt(celsius, 600.0f), 1e-4f) << celsius;
        EXPECT_NEAR(redness.ratio, rednessRatioAt(celsius, 600.0f), 1e-4f) << celsius;
    }
}

TEST(ColorModel, RampIsInsensitiveToTimestep){
    const CookState fine = simulateRamp(0.1f, 600.0f);
    const CookState coarse = simulateRamp(2.0f, 600.0f);
    EXPECT_LT(fine.redness.ratio, 0.95f);
    EXPECT_GT(fine.lightness.ratio, 1.2f);
    EXPECT_NEAR(coarse.redness.ratio, fine.redness.ratio, 2e-3f);
    EXPECT_NEAR(coarse.lightness.ratio, fine.lightness.ratio, 2e-3f);
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
    std::vector<float> schedule;
    for (int i = 0; i < 150; ++i) schedule.push_back(70.0f + 250.0f * static_cast<float>(i) / 149.0f);
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

    CookState state;
    for (float temperature : schedule) state = stepCook(state, temperature, dt);
    const glm::vec3 expected = cookedLab(glm::vec3(lab0), state);
    EXPECT_NEAR(solved.lab()[0].x, expected.x, 0.25f);
    EXPECT_NEAR(solved.lab()[0].y, expected.y, 0.25f);
    EXPECT_NEAR(solved.lab()[0].z, lab0.z, 0.08f);
    EXPECT_GT(solved.lab()[0].x, lab0.x + 5.0f);
    EXPECT_LT(solved.lab()[0].y, lab0.y - 0.5f);
    ASSERT_EQ(solved.cookRatio().size(), 2u);
    EXPECT_NEAR(solved.cookRatio()[0].x, state.lightness.ratio, 1e-4f);
    EXPECT_NEAR(solved.cookRatio()[0].y, state.redness.ratio, 1e-4f);
    EXPECT_EQ(solved.cookRatio()[1], glm::vec2(1.0f));
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
