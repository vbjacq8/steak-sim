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

TEST(ColorModel, DonenessTableInterpolation){
    EXPECT_EQ(donenessAt(39.9f).strength, 0.0f);
    EXPECT_EQ(donenessAt(kNoPeakCelsius).strength, 0.0f);
    EXPECT_EQ(donenessAt(std::nanf("")).strength, 0.0f);

    for (const DonenessKey& key : kDonenessKeys){
        const Doneness d = donenessAt(key.celsius);
        EXPECT_FLOAT_EQ(d.strength, key.strength) << key.celsius;
        EXPECT_FLOAT_EQ(d.lab.x, key.lab.x) << key.celsius;
        EXPECT_FLOAT_EQ(d.lab.y, key.lab.y) << key.celsius;
        EXPECT_FLOAT_EQ(d.lab.z, key.lab.z) << key.celsius;
    }

    const Doneness between = donenessAt(0.5f * (kDonenessKeys[2].celsius + kDonenessKeys[3].celsius));
    EXPECT_FLOAT_EQ(between.strength, 0.5f * (kDonenessKeys[2].strength + kDonenessKeys[3].strength));
    EXPECT_FLOAT_EQ(between.lab.x, 0.5f * (kDonenessKeys[2].lab.x + kDonenessKeys[3].lab.x));

    const DonenessKey& last = kDonenessKeys[kDonenessKeyCount - 1];
    const Doneness hot = donenessAt(400.0f);
    EXPECT_FLOAT_EQ(hot.strength, last.strength);
    EXPECT_FLOAT_EQ(hot.lab.x, last.lab.x);

    for (int i = 1; i < kDonenessKeyCount; ++i) EXPECT_GT(kDonenessKeys[i].celsius, kDonenessKeys[i - 1].celsius);
    // The zero-strength key shares the next key's color, so the blend never drifts toward another color.
    EXPECT_EQ(kDonenessKeys[0].lab, kDonenessKeys[1].lab);
}

TEST(ColorModel, CookedLabGoesFromRedToBrown){
    const glm::vec3 raw{52.8f, 41.2f, 27.0f};
    EXPECT_EQ(cookedLab(raw, kNoPeakCelsius), raw);
    EXPECT_EQ(cookedLab(raw, 30.0f), raw);

    const glm::vec3 rare = cookedLab(raw, fahrenheitToCelsius(120.0f));
    const glm::vec3 wellDone = cookedLab(raw, fahrenheitToCelsius(160.0f));
    const glm::vec3 crust = cookedLab(raw, 170.0f);
    EXPECT_LT(rare.y, raw.y);
    EXPECT_LT(wellDone.y, rare.y - 10.0f);
    EXPECT_GT(wellDone.x, rare.x);
    EXPECT_LT(crust.x, wellDone.x);
    EXPECT_LT(crust.y, wellDone.y);
    EXPECT_LT(channelError(crust, kDonenessKeys[kDonenessKeyCount - 1].lab), 1e-4f);
}

TEST(ColorModel, PeakOnlyRises){
    float peak = kNoPeakCelsius;
    for (float fahrenheit : {70.0f, 150.0f, 300.0f, 200.0f, 90.0f}) peak = raisePeak(peak, fahrenheit);
    EXPECT_FLOAT_EQ(peak, fahrenheitToCelsius(300.0f));
    EXPECT_FLOAT_EQ(raisePeak(peak, std::nanf("")), peak);
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

TEST(ColorSolver, GpuColorMatchesCpu){
    REQUIRE_VULKAN();
    const std::vector<uint8_t> image = texel(255, 0, 0);
    const glm::uvec3 dims{2, 1, 1};
    const std::vector<uint32_t> cells{CELL_SOLID, CELL_AIR};
    const std::vector<glm::vec2> uv{{0.25f, 0.25f}, {0.25f, 0.25f}};
    // Heats to 320 F (160 C), then cools; the color must keep the peak.
    std::vector<float> schedule;
    for (int i = 0; i < 100; ++i) schedule.push_back(70.0f + 250.0f * static_cast<float>(i) / 99.0f);
    for (int i = 1; i <= 50; ++i) schedule.push_back(320.0f - 220.0f * static_cast<float>(i) / 50.0f);
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

    float peak = kNoPeakCelsius;
    for (float temperature : schedule) peak = raisePeak(peak, temperature);
    const glm::vec3 expected = cookedLab(glm::vec3(lab0), peak);
    EXPECT_NEAR(solved.lab()[0].x, expected.x, 0.25f);
    EXPECT_NEAR(solved.lab()[0].y, expected.y, 0.25f);
    EXPECT_NEAR(solved.lab()[0].z, expected.z, 0.25f);
    EXPECT_LT(solved.lab()[0].x, lab0.x - 10.0f);
    EXPECT_LT(solved.lab()[0].y, 20.0f);
    ASSERT_EQ(solved.peakCelsius().size(), 2u);
    EXPECT_NEAR(solved.peakCelsius()[0], peak, 1e-3f);
    EXPECT_EQ(solved.peakCelsius()[1], kNoPeakCelsius);
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
