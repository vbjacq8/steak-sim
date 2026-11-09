#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// What a cell is to the solver. Must match CELL_* in shaders/compute/finitedifference.slang.
constexpr uint32_t CELL_AIR = 0u;    // held at HeatParams::airTemp; exchanges heat with solid neighbors through h
constexpr uint32_t CELL_SOLID = 1u;  // steps the heat equation
constexpr uint32_t CELL_PAN = 2u;    // held at HeatParams::panTemp; exchanges heat with solid neighbors through h

struct HeatParams{
    glm::uvec3 dims{128, 128, 128};
    float spacing = 1.0e-3f;        // cell edge length dx, m
    float timeStep = 1.0f;          // dt, s
    float diffusivity = 1.3e-7f;    // alpha, m^2/s (lean beef)
    float conductivity = 0.47f;     // k, W/(m K) (lean beef)
    float panTransfer = 400.0f;    // h between pan and meat, W/(m^2 K)
    float airTransfer = 20.0f;      // h between air and meat, W/(m^2 K); 10-30 is vapor around a cooking steak
    float initialTemp = 70.0f;      // F
    float panTemp = 400.0f;         // F
    float airTemp = 212.0f;          // F; raise toward 212 to model steam
    float duration = 240.0f;        // simulated time for the viewers, s; the solver takes a step count

    // r = alpha dt / dx^2.
    float fourier() const { return diffusivity * timeStep / (spacing * spacing); }

    // Reservoir-to-cell conductance relative to cell-to-cell conductance: the contact resistance 1/h in series
    // with conduction from the face to the cell center, dx / 2k. Tends to 2 as h grows.
    float faceCoupling(float transfer) const{
        const float biot = transfer * spacing / conductivity;
        return biot / (1.0f + 0.5f * biot);
    }
    float panCoupling() const { return faceCoupling(panTransfer); }
    float airCoupling() const { return faceCoupling(airTransfer); }

    // Largest stable r for the explicit scheme: a cell's own weight 1 - r * (sum of neighbor couplings)
    // must stay >= 0. A face is solid (weight 1), pan, or air. One pan face and five of the stronger of
    // solid and air covers a cell sitting on the pan.
    float maxStableFourier() const{
        const float side = std::max(1.0f, airCoupling());
        return 1.0f / std::max(6.0f * side, panCoupling() + 5.0f * side);
    }
};

struct HeatGrid{
    glm::uvec3 dims{};
    float spacing = 0.0f;
    float time = 0.0f;               // simulated seconds
    std::vector<float> temperature;  // index = x + nx * (y + ny * z)
    std::vector<uint32_t> cells;     // CELL_* kinds the field was solved with, same indexing

    size_t cellCount() const { return size_t(dims.x) * dims.y * dims.z; }
    size_t index(uint32_t x, uint32_t y, uint32_t z) const { return x + dims.x * (y + size_t(dims.y) * z); }
    float at(uint32_t x, uint32_t y, uint32_t z) const { return temperature[index(x, y, z)]; }
    uint32_t kind(uint32_t x, uint32_t y, uint32_t z) const { return cells[index(x, y, z)]; }
};

// Cell kinds for a grid that is CELL_SOLID where `solid` is nonzero (e.g. VoxelGrid::cells) and CELL_AIR
// elsewhere, except that the whole y = 0 layer is the pan.
std::vector<uint32_t> makePanCells(glm::uvec3 dims, const std::vector<uint32_t>& solid);

// A grid that is solid everywhere, with the y = 0 layer as the pan.
std::vector<uint32_t> makeBlockCells(glm::uvec3 dims);

// Runs shaders/compute/finitedifference.slang. The device, physical device and queue must outlive it.
// Every run throws std::invalid_argument if params.fourier() > params.maxStableFourier() (unstable) or the inputs
// don't match params.dims.
class HeatSolver{
public:
    HeatSolver(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
               const vk::raii::Queue& queue, uint32_t queueFamilyIndex, const std::string& shaderPath);

    // makeBlockCells(params.dims), starting from initialTemp with pan cells at panTemp and air cells at airTemp.
    HeatGrid run(const HeatParams& params, uint32_t steps);

    // `cells` holds a CELL_* kind per cell. Starts from initialTemp, with pan cells at panTemp and air cells at airTemp.
    HeatGrid run(const HeatParams& params, const std::vector<uint32_t>& cells, uint32_t steps);

    // Starts from an arbitrary field (params.initialTemp is ignored). Pan cells become panTemp, and air cells
    // become airTemp, after the first step.
    HeatGrid run(const HeatParams& params, const std::vector<uint32_t>& cells, const std::vector<float>& initial,
                 uint32_t steps);

private:
    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                                   vk::MemoryPropertyFlags properties) const;

    const vk::raii::Device& device;
    const vk::raii::PhysicalDevice& physicalDevice;
    const vk::raii::Queue& queue;
    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::DescriptorSetLayout setLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;
};
