#include "compute_context.hpp"
#include "voxelizer.hpp"

#include <gtest/gtest.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <random>

namespace {

using Triangle = std::array<glm::vec3, 3>;
static_assert(sizeof(Triangle) == 9 * sizeof(float));

// ---------------------------------------------------------------------------
// Headless Vulkan
// ---------------------------------------------------------------------------

// Shared by all tests; nullptr if no usable Vulkan device, in which case GPU tests skip.
ComputeContext* vulkan(){
    static std::unique_ptr<ComputeContext> ctx = []() -> std::unique_ptr<ComputeContext>{
        try{
            return std::make_unique<ComputeContext>("voxel_tests");
        } catch (const std::exception& e){
            std::cerr << "Vulkan unavailable: " << e.what() << std::endl;
            return nullptr;
        }
    }();
    return ctx.get();
}

#define REQUIRE_VULKAN() \
    if (vulkan() == nullptr) GTEST_SKIP() << "no usable Vulkan device"

struct HostBuffer{
    vk::raii::Buffer buffer = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    void* mapped = nullptr;
};

HostBuffer createHostBuffer(ComputeContext& ctx, vk::DeviceSize size){
    HostBuffer result;
    result.buffer = vk::raii::Buffer(ctx.device, vk::BufferCreateInfo{
        .size = size, .usage = vk::BufferUsageFlagBits::eStorageBuffer, .sharingMode = vk::SharingMode::eExclusive});
    const vk::MemoryRequirements requirements = result.buffer.getMemoryRequirements();
    const vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const vk::PhysicalDeviceMemoryProperties memoryProperties = ctx.physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i){
        if ((requirements.memoryTypeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties){
            result.memory = vk::raii::DeviceMemory(ctx.device, vk::MemoryAllocateInfo{
                .allocationSize = requirements.size, .memoryTypeIndex = i});
            result.buffer.bindMemory(*result.memory, 0);
            result.mapped = result.memory.mapMemory(0, size);
            return result;
        }
    }
    throw std::runtime_error("no host-visible memory type");
}

// ---------------------------------------------------------------------------
// GPU kernel around voxel_math.slang (tests/voxel_tests.slang)
// ---------------------------------------------------------------------------

// Must match KernelParams in voxel_tests.slang.
struct KernelParams{
    glm::vec3 point;
    uint32_t triangleCount;
    glm::vec3 boxHalfSize;
    uint32_t mode;
};
static_assert(sizeof(KernelParams) == 32);

class TriangleKernel{
public:
    enum Mode : uint32_t { TriangleBox = 0, RayCrossesUp = 1 };

    explicit TriangleKernel(ComputeContext& ctx) : ctx(ctx){
        commandPool = vk::raii::CommandPool(ctx.device, vk::CommandPoolCreateInfo{.queueFamilyIndex = ctx.queueFamily});

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{};
        for (uint32_t i = 0; i < bindings.size(); ++i){
            bindings[i] = {.binding = i, .descriptorType = vk::DescriptorType::eStorageBuffer,
                           .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute};
        }
        setLayout = vk::raii::DescriptorSetLayout(ctx.device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});
        vk::PushConstantRange pushConstantRange{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(KernelParams)};
        pipelineLayout = vk::raii::PipelineLayout(ctx.device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1, .pSetLayouts = &*setLayout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange});

        std::ifstream file(VOXEL_TESTS_SPV, std::ios::ate | std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("failed to open " VOXEL_TESTS_SPV);
        std::vector<char> code(file.tellg());
        file.seekg(0);
        file.read(code.data(), static_cast<std::streamsize>(code.size()));
        vk::raii::ShaderModule module(ctx.device, vk::ShaderModuleCreateInfo{
            .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data())});
        pipeline = vk::raii::Pipeline(ctx.device, nullptr, vk::ComputePipelineCreateInfo{
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "testMain"},
            .layout = pipelineLayout});

        vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 2};
        descriptorPool = vk::raii::DescriptorPool(ctx.device, vk::DescriptorPoolCreateInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &poolSize});
    }

    // One 0/1 result per triangle. Records `repeats` back-to-back dispatches over the same
    // input; if `seconds` is given it receives the wall time from submit until the queue is idle.
    std::vector<uint32_t> run(const std::vector<Triangle>& triangles, Mode mode, glm::vec3 point,
                              glm::vec3 boxHalfSize = glm::vec3(0.5f), int repeats = 1, double* seconds = nullptr){
        const uint32_t count = static_cast<uint32_t>(triangles.size());
        HostBuffer input = createHostBuffer(ctx, triangles.size() * sizeof(Triangle));
        HostBuffer output = createHostBuffer(ctx, triangles.size() * sizeof(uint32_t));
        std::memcpy(input.mapped, triangles.data(), triangles.size() * sizeof(Triangle));

        vk::raii::DescriptorSet descriptorSet = std::move(ctx.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
            .descriptorPool = descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &*setLayout}).front());
        std::array<vk::DescriptorBufferInfo, 2> bufferInfos{{
            {.buffer = input.buffer, .offset = 0, .range = vk::WholeSize},
            {.buffer = output.buffer, .offset = 0, .range = vk::WholeSize},
        }};
        std::array<vk::WriteDescriptorSet, 2> writes{};
        for (uint32_t i = 0; i < writes.size(); ++i){
            writes[i] = {.dstSet = descriptorSet, .dstBinding = i, .descriptorCount = 1,
                         .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &bufferInfos[i]};
        }
        ctx.device.updateDescriptorSets(writes, {});

        const KernelParams params{.point = point, .triangleCount = count, .boxHalfSize = boxHalfSize, .mode = mode};
        vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(ctx.device, vk::CommandBufferAllocateInfo{
            .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1}).front());
        commandBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, *descriptorSet, nullptr);
        commandBuffer.pushConstants<KernelParams>(*pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
        const vk::MemoryBarrier2 betweenRuns{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader, .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader, .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite};
        for (int r = 0; r < repeats; ++r){
            if (r > 0) commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &betweenRuns});
            commandBuffer.dispatch((count + 63) / 64, 1, 1);
        }
        const vk::MemoryBarrier2 toHost{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader, .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost, .dstAccessMask = vk::AccessFlagBits2::eHostRead};
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
        commandBuffer.end();

        const auto start = std::chrono::steady_clock::now();
        ctx.queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer}, nullptr);
        ctx.queue.waitIdle();
        if (seconds) *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        std::vector<uint32_t> results(count);
        std::memcpy(results.data(), output.mapped, count * sizeof(uint32_t));
        return results;
    }

private:
    ComputeContext& ctx;
    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::DescriptorSetLayout setLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
};

// ---------------------------------------------------------------------------
// CPU references (independent algorithms, double precision)
// ---------------------------------------------------------------------------

// Clips the triangle against the box's six planes; anything left over means overlap.
bool clipOverlapReference(const Triangle& tri, glm::dvec3 center, glm::dvec3 halfSize){
    std::vector<glm::dvec3> polygon = {glm::dvec3(tri[0]) - center, glm::dvec3(tri[1]) - center, glm::dvec3(tri[2]) - center};
    for (int axis = 0; axis < 3; ++axis){
        for (double sign : {1.0, -1.0}){
            std::vector<glm::dvec3> clipped;
            for (size_t i = 0; i < polygon.size(); ++i){
                const glm::dvec3& p = polygon[i];
                const glm::dvec3& q = polygon[(i + 1) % polygon.size()];
                const double dp = sign * p[axis] - halfSize[axis];
                const double dq = sign * q[axis] - halfSize[axis];
                if (dp <= 0.0) clipped.push_back(p);
                if ((dp < 0.0 && dq > 0.0) || (dp > 0.0 && dq < 0.0)) clipped.push_back(p + (q - p) * (dp / (dp - dq)));
            }
            polygon = std::move(clipped);
            if (polygon.empty()) return false;
        }
    }
    return true;
}

// nullopt when the answer flips between a slightly shrunk and a slightly grown box,
// i.e. float rounding on the GPU could legitimately go either way.
std::optional<bool> triangleBoxReference(const Triangle& tri, glm::dvec3 center, glm::dvec3 halfSize){
    const bool grown = clipOverlapReference(tri, center, halfSize * (1.0 + 1e-4));
    const bool shrunk = clipOverlapReference(tri, center, halfSize * (1.0 - 1e-4));
    if (grown != shrunk) return std::nullopt;
    return grown;
}

// Möller-Trumbore with direction +Y. nullopt when the ray passes within `eps` of an
// edge, starts within `eps` of the plane, or the triangle is nearly edge-on.
std::optional<bool> rayCrossesUpReference(const Triangle& tri, glm::dvec3 origin, double eps = 1e-6){
    const glm::dvec3 dir(0.0, 1.0, 0.0);
    const glm::dvec3 a(tri[0]), b(tri[1]), c(tri[2]);
    const glm::dvec3 e1 = b - a, e2 = c - a;
    const glm::dvec3 p = glm::cross(dir, e2);
    const double det = glm::dot(e1, p);
    if (std::abs(det) < eps) return std::nullopt;
    const glm::dvec3 s = origin - a;
    const glm::dvec3 q = glm::cross(s, e1);
    const double u = glm::dot(s, p) / det;
    const double v = glm::dot(dir, q) / det;
    const double w = 1.0 - u - v;
    const double t = glm::dot(e2, q) / det;
    if (std::min({std::abs(u), std::abs(v), std::abs(w), std::abs(t)}) < eps) return std::nullopt;
    return u > 0.0 && v > 0.0 && w > 0.0 && t > 0.0;
}

// Voorhies' setup: vertices uniform in a 4x4x4 cube centered on the origin.
std::vector<Triangle> randomTriangles(size_t count, uint32_t seed){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> coordinate(-2.0f, 2.0f);
    std::vector<Triangle> triangles(count);
    for (Triangle& tri : triangles){
        for (glm::vec3& v : tri) v = {coordinate(rng), coordinate(rng), coordinate(rng)};
    }
    return triangles;
}

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

struct Mesh{
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
};

Mesh boxMesh(glm::vec3 lo, glm::vec3 hi){
    Mesh mesh;
    for (uint32_t i = 0; i < 8; ++i){
        mesh.positions.push_back({(i & 1) ? hi.x : lo.x, (i & 2) ? hi.y : lo.y, (i & 4) ? hi.z : lo.z});
    }
    const uint32_t quads[6][4] = {{0, 2, 6, 4}, {1, 5, 7, 3}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 6, 7, 5}};
    for (const auto& q : quads){
        mesh.indices.insert(mesh.indices.end(), {q[0], q[1], q[2], q[0], q[2], q[3]});
    }
    return mesh;
}

Mesh uvSphere(float radius, uint32_t stacks, uint32_t slices){
    const float pi = 3.14159265358979f;
    Mesh mesh;
    mesh.positions.push_back({0.0f, radius, 0.0f});
    for (uint32_t i = 1; i < stacks; ++i){
        const float phi = pi * float(i) / float(stacks);
        for (uint32_t j = 0; j < slices; ++j){
            const float theta = 2.0f * pi * float(j) / float(slices);
            mesh.positions.push_back(radius * glm::vec3(std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta)));
        }
    }
    mesh.positions.push_back({0.0f, -radius, 0.0f});
    const uint32_t south = static_cast<uint32_t>(mesh.positions.size() - 1);
    const auto ring = [slices](uint32_t i, uint32_t j){ return 1 + (i - 1) * slices + (j % slices); };

    for (uint32_t j = 0; j < slices; ++j){
        mesh.indices.insert(mesh.indices.end(), {0, ring(1, j + 1), ring(1, j)});
        mesh.indices.insert(mesh.indices.end(), {south, ring(stacks - 1, j), ring(stacks - 1, j + 1)});
    }
    for (uint32_t i = 1; i + 1 < stacks; ++i){
        for (uint32_t j = 0; j < slices; ++j){
            const uint32_t a = ring(i, j), b = ring(i, j + 1), c = ring(i + 1, j + 1), d = ring(i + 1, j);
            mesh.indices.insert(mesh.indices.end(), {a, b, c, a, c, d});
        }
    }
    return mesh;
}

Mesh loadObj(const std::string& path){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) throw std::runtime_error(warn + err);

    Mesh mesh;
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3){
        mesh.positions.push_back({attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]});
    }
    for (const auto& shape : shapes){
        for (const auto& index : shape.mesh.indices) mesh.indices.push_back(static_cast<uint32_t>(index.vertex_index));
    }
    return mesh;
}

// Exact enclosed volume of a closed, consistently wound mesh (divergence theorem).
double meshVolume(const Mesh& mesh){
    double sixVolume = 0.0;
    for (size_t t = 0; t + 2 < mesh.indices.size(); t += 3){
        const glm::dvec3 a(mesh.positions[mesh.indices[t]]);
        const glm::dvec3 b(mesh.positions[mesh.indices[t + 1]]);
        const glm::dvec3 c(mesh.positions[mesh.indices[t + 2]]);
        sixVolume += glm::dot(a, glm::cross(b, c));
    }
    return std::abs(sixVolume) / 6.0;
}

size_t countFlag(const VoxelGrid& grid, uint32_t flag){
    return static_cast<size_t>(std::count_if(grid.cells.begin(), grid.cells.end(), [flag](uint32_t c){ return (c & flag) != 0; }));
}

glm::vec3 voxelCenter(const VoxelGrid& grid, uint32_t x, uint32_t y, uint32_t z){
    return grid.origin + (glm::vec3(x, y, z) + 0.5f) * grid.voxelSize;
}

Voxelizer makeVoxelizer(){
    ComputeContext& ctx = *vulkan();
    return Voxelizer(ctx.device, ctx.physicalDevice, ctx.queue, ctx.queueFamily, VOXELIZE_SPV);
}

// ---------------------------------------------------------------------------
// Grid layout (CPU only)
// ---------------------------------------------------------------------------

TEST(VoxelGridLayout, SnapsToVoxelMultiplesWithOneVoxelPadding){
    const VoxelGrid grid = makeVoxelGridLayout({-16.21f, 0.0f, -9.915f}, {16.144f, 5.067f, 9.338f}, 0.5f);
    EXPECT_EQ(grid.dims, glm::uvec3(68, 13, 41));
    EXPECT_FLOAT_EQ(grid.origin.x, -17.0f);
    EXPECT_FLOAT_EQ(grid.origin.y, -0.5f);
    EXPECT_FLOAT_EQ(grid.origin.z, -10.5f);
    EXPECT_FLOAT_EQ(grid.voxelSize, 0.5f);
    EXPECT_TRUE(grid.cells.empty());
}

TEST(VoxelGridLayout, BoundsOnVoxelBoundariesStillGetPadding){
    const VoxelGrid grid = makeVoxelGridLayout(glm::vec3(0.0f), glm::vec3(1.0f), 0.25f);
    EXPECT_EQ(grid.dims, glm::uvec3(6));
    EXPECT_FLOAT_EQ(grid.origin.x, -0.25f);
    EXPECT_EQ(grid.cellCount(), 216u);
}

// ---------------------------------------------------------------------------
// triangleBoxOverlap (GPU)
// ---------------------------------------------------------------------------

struct OverlapCase{
    const char* name;
    Triangle triangle;
    bool expected;
};

const std::vector<OverlapCase> OVERLAP_CASES = {
    {"fully inside", {{{-0.1f, -0.1f, 0.0f}, {0.1f, -0.1f, 0.0f}, {0.0f, 0.1f, 0.0f}}}, true},
    {"far away", {{{5.0f, 5.0f, 5.0f}, {6.0f, 5.0f, 5.0f}, {5.0f, 6.0f, 5.0f}}}, false},
    {"cuts through box with every vertex outside", {{{-5.0f, -5.0f, 0.0f}, {5.0f, -5.0f, 0.0f}, {0.0f, 5.0f, 0.0f}}}, true},
    {"box corner pokes through triangle", {{{3.0f, -0.8f, -0.8f}, {-0.8f, 3.0f, -0.8f}, {-0.8f, -0.8f, 3.0f}}}, true},
    {"separated by triangle plane only", {{{1.6f, 0.0f, 0.0f}, {0.0f, 1.6f, 0.0f}, {0.0f, 0.0f, 1.6f}}}, false},
    {"separated by edge cross axis only", {{{1.1f, 0.0f, 0.0f}, {0.0f, 1.1f, 0.0f}, {1.5f, 1.5f, 0.0f}}}, false},
    {"vertex touching box face", {{{0.5f, 0.0f, 0.0f}, {2.0f, 1.0f, 0.0f}, {2.0f, -1.0f, 0.0f}}}, true},
    {"degenerate segment through box", {{{-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}}, true},
    {"degenerate segment outside box", {{{-1.0f, 2.0f, 0.0f}, {1.0f, 2.0f, 0.0f}, {0.0f, 2.0f, 0.0f}}}, false},
};

TEST(TriangleBoxOverlap, ReferenceAgreesWithHandPickedCases){
    for (const OverlapCase& c : OVERLAP_CASES){
        SCOPED_TRACE(c.name);
        EXPECT_EQ(clipOverlapReference(c.triangle, glm::dvec3(0.0), glm::dvec3(0.5)), c.expected);
    }
}

TEST(TriangleBoxOverlap, HandPickedCases){
    REQUIRE_VULKAN();
    TriangleKernel kernel(*vulkan());

    // Same cases around the origin and around a box moved far away, to catch translation bugs.
    for (const glm::vec3 offset : {glm::vec3(0.0f), glm::vec3(10.0f, -3.0f, 2.0f)}){
        std::vector<Triangle> triangles;
        for (const OverlapCase& c : OVERLAP_CASES){
            triangles.push_back({c.triangle[0] + offset, c.triangle[1] + offset, c.triangle[2] + offset});
        }
        const std::vector<uint32_t> results = kernel.run(triangles, TriangleKernel::TriangleBox, offset);
        for (size_t i = 0; i < OVERLAP_CASES.size(); ++i){
            SCOPED_TRACE(std::string(OVERLAP_CASES[i].name) + " at offset (" + std::to_string(offset.x) + ", " +
                         std::to_string(offset.y) + ", " + std::to_string(offset.z) + ")");
            EXPECT_EQ(results[i] != 0, OVERLAP_CASES[i].expected);
        }
    }
}

TEST(TriangleBoxOverlap, MatchesClippingReferenceOnRandomTriangles){
    REQUIRE_VULKAN();
    TriangleKernel kernel(*vulkan());
    const std::vector<Triangle> triangles = randomTriangles(100'000, 1234);
    const std::vector<uint32_t> results = kernel.run(triangles, TriangleKernel::TriangleBox, glm::vec3(0.0f));

    size_t mismatches = 0, ambiguous = 0;
    for (size_t i = 0; i < triangles.size(); ++i){
        const std::optional<bool> expected = triangleBoxReference(triangles[i], glm::dvec3(0.0), glm::dvec3(0.5));
        if (!expected){
            ++ambiguous;
            continue;
        }
        if ((results[i] != 0) != *expected && ++mismatches <= 5){
            ADD_FAILURE() << "triangle " << i << ": GPU " << results[i] << ", reference " << *expected;
        }
    }
    EXPECT_EQ(mismatches, 0u);
    EXPECT_LT(ambiguous, triangles.size() / 1000) << "too many near-boundary cases to be meaningful";
}

// Akenine-Möller's performance test, after Voorhies: 100,000 random triangles with vertices
// in a 4x4x4 cube around the origin, tested against the unit cube 100 times in a row.
TEST(TriangleBoxOverlap, VoorhiesBenchmark){
    REQUIRE_VULKAN();
    constexpr size_t TRIANGLES = 100'000;
    constexpr int REPEATS = 100;
    TriangleKernel kernel(*vulkan());
    const std::vector<Triangle> triangles = randomTriangles(TRIANGLES, 42);

    const std::vector<uint32_t> once = kernel.run(triangles, TriangleKernel::TriangleBox, glm::vec3(0.0f));  // also warms up
    double seconds = 0.0;
    const std::vector<uint32_t> repeated = kernel.run(triangles, TriangleKernel::TriangleBox, glm::vec3(0.0f),
                                                      glm::vec3(0.5f), REPEATS, &seconds);
    EXPECT_EQ(once, repeated);

    const double tests = double(TRIANGLES) * REPEATS;
    const double overlapFraction = double(std::count(once.begin(), once.end(), 1u)) / TRIANGLES;
    std::cout << "[ BENCHMARK] " << TRIANGLES << " triangles x " << REPEATS << " runs: "
              << seconds * 1e3 << " ms total, " << seconds * 1e9 / tests << " ns/test, "
              << overlapFraction * 100.0 << "% overlapping" << std::endl;
    RecordProperty("total_ms", std::to_string(seconds * 1e3));
    RecordProperty("ns_per_test", std::to_string(seconds * 1e9 / tests));
    RecordProperty("overlap_percent", std::to_string(overlapFraction * 100.0));
}

// ---------------------------------------------------------------------------
// rayCrossesUp (GPU)
// ---------------------------------------------------------------------------

TEST(RayCrossesUp, HandPickedCases){
    REQUIRE_VULKAN();
    TriangleKernel kernel(*vulkan());
    struct Case{ const char* name; Triangle triangle; bool expected; };
    const std::vector<Case> cases = {
        {"flat triangle above", {{{-1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, -1.0f}, {0.0f, 1.0f, 1.0f}}}, true},
        {"flat triangle below", {{{-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}, {0.0f, -1.0f, 1.0f}}}, false},
        {"tilted triangle above", {{{-1.0f, 0.5f, -1.0f}, {1.0f, 1.5f, -1.0f}, {0.0f, 2.0f, 1.0f}}}, true},
        {"beside the ray", {{{4.0f, 1.0f, -1.0f}, {6.0f, 1.0f, -1.0f}, {5.0f, 1.0f, 1.0f}}}, false},
        {"edge-on (vertical) triangle", {{{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {0.0f, 3.0f, 0.0f}}}, false},
    };
    std::vector<Triangle> triangles;
    for (const Case& c : cases) triangles.push_back(c.triangle);
    const std::vector<uint32_t> results = kernel.run(triangles, TriangleKernel::RayCrossesUp, glm::vec3(0.1f, 0.0f, 0.05f));
    for (size_t i = 0; i < cases.size(); ++i){
        SCOPED_TRACE(cases[i].name);
        EXPECT_EQ(results[i] != 0, cases[i].expected);
    }
}

TEST(RayCrossesUp, MatchesMollerTrumboreOnRandomTriangles){
    REQUIRE_VULKAN();
    TriangleKernel kernel(*vulkan());
    const std::vector<Triangle> triangles = randomTriangles(100'000, 99);

    for (const glm::vec3 origin : {glm::vec3(0.1234f, -0.2f, 0.0567f), glm::vec3(-1.3f, 0.7f, 0.9f)}){
        const std::vector<uint32_t> results = kernel.run(triangles, TriangleKernel::RayCrossesUp, origin);
        size_t mismatches = 0, ambiguous = 0;
        for (size_t i = 0; i < triangles.size(); ++i){
            const std::optional<bool> expected = rayCrossesUpReference(triangles[i], glm::dvec3(origin));
            if (!expected){
                ++ambiguous;
                continue;
            }
            if ((results[i] != 0) != *expected && ++mismatches <= 5){
                ADD_FAILURE() << "triangle " << i << ": GPU " << results[i] << ", reference " << *expected;
            }
        }
        EXPECT_EQ(mismatches, 0u);
        EXPECT_LT(ambiguous, triangles.size() / 1000);
    }
}

// ---------------------------------------------------------------------------
// Voxelizer end to end (GPU)
// ---------------------------------------------------------------------------

TEST(Voxelizer, RejectsEmptyMesh){
    REQUIRE_VULKAN();
    Voxelizer voxelizer = makeVoxelizer();
    EXPECT_THROW(voxelizer.voxelize({}, {}, 0.25f), std::invalid_argument);
}

// Box from 0.05 to 0.95 with 0.25 voxels: voxel centers 0.125..0.875 fall inside on
// indices 1..4 of each axis, and only the outer layer of that 4x4x4 block touches a face.
TEST(Voxelizer, AxisAlignedBoxHasExactCounts){
    REQUIRE_VULKAN();
    Voxelizer voxelizer = makeVoxelizer();
    const Mesh box = boxMesh(glm::vec3(0.05f), glm::vec3(0.95f));
    const VoxelGrid grid = voxelizer.voxelize(box.positions, box.indices, 0.25f);

    ASSERT_EQ(grid.dims, glm::uvec3(6));
    EXPECT_EQ(countFlag(grid, VOXEL_INSIDE), 64u);
    EXPECT_EQ(countFlag(grid, VOXEL_SURFACE), 56u);
    EXPECT_EQ(countFlag(grid, VOXEL_INSIDE | VOXEL_SURFACE), 64u);
    for (uint32_t z = 0; z < 6; ++z){
        for (uint32_t y = 0; y < 6; ++y){
            for (uint32_t x = 0; x < 6; ++x){
                const auto inBlock = [](uint32_t i){ return i >= 1 && i <= 4; };
                const auto onLayer = [](uint32_t i){ return i == 1 || i == 4; };
                const bool inside = inBlock(x) && inBlock(y) && inBlock(z);
                const bool surface = inside && (onLayer(x) || onLayer(y) || onLayer(z));
                SCOPED_TRACE("voxel (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")");
                EXPECT_EQ((grid.at(x, y, z) & VOXEL_INSIDE) != 0, inside);
                EXPECT_EQ((grid.at(x, y, z) & VOXEL_SURFACE) != 0, surface);
            }
        }
    }
}

TEST(Voxelizer, TranslationByWholeVoxelsGivesIdenticalCells){
    REQUIRE_VULKAN();
    Voxelizer voxelizer = makeVoxelizer();
    const Mesh box = boxMesh(glm::vec3(0.05f), glm::vec3(0.95f));
    const Mesh moved = boxMesh(glm::vec3(0.05f) + glm::vec3(2.0f, -1.0f, 3.5f), glm::vec3(0.95f) + glm::vec3(2.0f, -1.0f, 3.5f));
    const VoxelGrid a = voxelizer.voxelize(box.positions, box.indices, 0.25f);
    const VoxelGrid b = voxelizer.voxelize(moved.positions, moved.indices, 0.25f);
    EXPECT_EQ(a.dims, b.dims);
    EXPECT_EQ(a.cells, b.cells);
    EXPECT_FLOAT_EQ(b.origin.x - a.origin.x, 2.0f);
}

TEST(Voxelizer, SphereInsideMatchesMeshVolumeAndShape){
    REQUIRE_VULKAN();
    Voxelizer voxelizer = makeVoxelizer();
    const Mesh sphere = uvSphere(1.0f, 24, 48);
    const VoxelGrid grid = voxelizer.voxelize(sphere.positions, sphere.indices, 0.05f);

    const double voxelVolume = std::pow(double(grid.voxelSize), 3);
    const double insideVolume = double(countFlag(grid, VOXEL_INSIDE)) * voxelVolume;
    EXPECT_NEAR(insideVolume, meshVolume(sphere), 0.02 * meshVolume(sphere));

    // Well inside the facets must be inside; well outside must not be.
    for (uint32_t z = 0; z < grid.dims.z; ++z){
        for (uint32_t y = 0; y < grid.dims.y; ++y){
            for (uint32_t x = 0; x < grid.dims.x; ++x){
                const float r = glm::length(voxelCenter(grid, x, y, z));
                const bool inside = (grid.at(x, y, z) & VOXEL_INSIDE) != 0;
                if (r < 0.9f) ASSERT_TRUE(inside) << "voxel at radius " << r;
                if (r > 1.1f) ASSERT_FALSE(inside) << "voxel at radius " << r;
            }
        }
    }
}

// Wherever an inside voxel neighbors an outside one, the surface passes through one of the
// two boxes, so one of them must be flagged; a miss would be a hole in the surface.
TEST(Voxelizer, SurfaceHasNoGapsBetweenInsideAndOutside){
    REQUIRE_VULKAN();
    Voxelizer voxelizer = makeVoxelizer();
    const Mesh sphere = uvSphere(1.0f, 24, 48);
    const VoxelGrid grid = voxelizer.voxelize(sphere.positions, sphere.indices, 0.05f);

    size_t gaps = 0;
    for (uint32_t z = 0; z + 1 < grid.dims.z; ++z){
        for (uint32_t y = 0; y + 1 < grid.dims.y; ++y){
            for (uint32_t x = 0; x + 1 < grid.dims.x; ++x){
                const uint32_t cell = grid.at(x, y, z);
                for (const uint32_t neighbor : {grid.at(x + 1, y, z), grid.at(x, y + 1, z), grid.at(x, y, z + 1)}){
                    const bool boundary = ((cell ^ neighbor) & VOXEL_INSIDE) != 0;
                    const bool flagged = ((cell | neighbor) & VOXEL_SURFACE) != 0;
                    gaps += boundary && !flagged;
                }
            }
        }
    }
    EXPECT_EQ(gaps, 0u);
}

TEST(Voxelizer, SteakInsideMatchesMeshVolume){
    REQUIRE_VULKAN();
    Voxelizer voxelizer = makeVoxelizer();
    const Mesh steak = loadObj(STEAK_OBJ);
    const VoxelGrid grid = voxelizer.voxelize(steak.positions, steak.indices, 0.5f);

    EXPECT_EQ(grid.dims, glm::uvec3(68, 13, 41));
    const double insideVolume = double(countFlag(grid, VOXEL_INSIDE)) * std::pow(double(grid.voxelSize), 3);
    EXPECT_NEAR(insideVolume, meshVolume(steak), 0.02 * meshVolume(steak));
}

}  // namespace
