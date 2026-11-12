#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// What a cell is to the solver. Must match CELL_* in shaders/compute/finitedifference.slang.
constexpr uint32_t CELL_AIR = 0u;    // held at HeatParams::airTemp; exchanges heat with solid neighbors through h
constexpr uint32_t CELL_SOLID = 1u;  // steps the heat equation
constexpr uint32_t CELL_PAN = 2u;    // held at HeatParams::panTemp; exchanges heat with solid neighbors through h

struct HeatParams{
    glm::uvec3 dims{128, 128, 128};
    float spacing = 3.0e-3f;        // cell edge length dx, m
    float timeStep = 1.0f;          // dt, s
    float diffusivity = 1.3e-7f;    // alpha, m^2/s (lean beef)
    float conductivity = 0.47f;     // k, W/(m K) (lean beef)
    float panTransfer = 400.0f;    // h between pan and meat, W/(m^2 K)
    float airTransfer = 20.0f;      // h between air and meat, W/(m^2 K); 10-30 is vapor around a cooking steak
    float initialTemp = 70.0f;      // F
    float panTemp = 400.0f;         // F
    float airTemp = 212.0f;          // F; raise toward 212 to model steam
    float duration = 240.0f;        // simulated time for the viewers, s; the solver takes a step count
    uint32_t panAxis = 2;           // makePanCells puts the pan in the low layer along this axis (0 x, 1 y, 2 z)

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
    uint32_t panAxis = 2;            // HeatParams::panAxis of the run; tells the viewer which side is the pan
    std::vector<float> temperature;  // degrees Fahrenheit; index = x + nx * (y + ny * z)
    std::vector<uint32_t> cells;     // CELL_* kinds the field was solved with, same indexing
    // Linear sRGB from the color simulation, one entry per cell. Empty when color is not run;
    // the viewer then keeps the temperature colors. Zero for cells that are not meat.
    std::vector<glm::vec3> color;
    // Linear sRGB of the albedo sample before cooking. Same indexing as color; empty when color is not run.
    std::vector<glm::vec3> initialColor;
    // Hottest temperature each cell reached, Celsius (kNoPeakCelsius in color_model.hpp where none). Lets
    // the viewer cook the full-resolution albedo the same way. Empty when color is not run; same indexing.
    std::vector<float> peakCelsius;
    // Median luminance (CIE Y) and a* of the albedo's meat, the reference that cooked the color
    // (ColorReference in color_model.hpp). 0 when color is not run.
    float colorReferenceY = 0.0f;
    float colorReferenceA = 0.0f;
    // UV that seeded each cell from the albedo. (-1, -1) when the cell has none. Empty if color
    // was not run. Same indexing as color.
    std::vector<glm::vec2> sampleUv;

    size_t cellCount() const { return size_t(dims.x) * dims.y * dims.z; }
    size_t index(uint32_t x, uint32_t y, uint32_t z) const { return x + dims.x * (y + size_t(dims.y) * z); }
    float at(uint32_t x, uint32_t y, uint32_t z) const { return temperature[index(x, y, z)]; }
    uint32_t kind(uint32_t x, uint32_t y, uint32_t z) const { return cells[index(x, y, z)]; }
};

// Cell kinds for a grid that is CELL_SOLID where `solid` is nonzero (e.g. VoxelGrid::cells) and CELL_AIR
// elsewhere, except that the whole low layer along `panAxis` (z = 0 by default) is the pan.
std::vector<uint32_t> makePanCells(glm::uvec3 dims, const std::vector<uint32_t>& solid, uint32_t panAxis = 2);

// A grid that is solid everywhere, with the z = 0 layer as the pan.
std::vector<uint32_t> makeBlockCells(glm::uvec3 dims);

// Optional GPU work recorded in the same command buffers as the heat steps.
// prepare() runs after the ping-pong temperature buffers exist and before they are recorded.
// recordInit() runs once the initial temperature is visible to compute. recordAfterStep() runs after
// step `stepIndex` (0-based) has written its temperature; that temperature is in buffer (stepIndex + 1) % 2
// of the pair passed to prepare(). recordReadback() runs after the last step. The queue is idle when run()
// returns, so a buffer written by recordReadback can be mapped.
class HeatStepHook{
public:
    virtual ~HeatStepHook() = default;
    virtual void prepare(vk::Buffer temperature0, vk::Buffer temperature1) = 0;
    virtual void recordInit(vk::raii::CommandBuffer& commandBuffer) = 0;
    virtual void recordAfterStep(vk::raii::CommandBuffer& commandBuffer, uint32_t stepIndex) = 0;
    virtual void recordReadback(vk::raii::CommandBuffer& commandBuffer) = 0;
};

// Runs shaders/compute/finitedifference.slang. The device, physical device and queue must outlive it.
// Every run throws std::invalid_argument if params.fourier() > params.maxStableFourier() (unstable) or the inputs
// don't match params.dims. `hook`, when set, runs once per heat step after that step's temperature is visible.
class HeatSolver{
public:
    HeatSolver(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
               const vk::raii::Queue& queue, uint32_t queueFamilyIndex, const std::string& shaderPath);

    // makeBlockCells(params.dims), starting from initialTemp with pan cells at panTemp and air cells at airTemp.
    // `progress`, when set, is called with the fraction of steps finished (0, 1] after each batch of submits.
    HeatGrid run(const HeatParams& params, uint32_t steps, const std::function<void(float)>& progress = {},
                 HeatStepHook* hook = nullptr);

    // `cells` holds a CELL_* kind per cell. Starts from initialTemp, with pan cells at panTemp and air cells at airTemp.
    HeatGrid run(const HeatParams& params, const std::vector<uint32_t>& cells, uint32_t steps,
                 const std::function<void(float)>& progress = {}, HeatStepHook* hook = nullptr);

    // Starts from an arbitrary field (params.initialTemp is ignored). Pan cells become panTemp, and air cells
    // become airTemp, after the first step.
    HeatGrid run(const HeatParams& params, const std::vector<uint32_t>& cells, const std::vector<float>& initial,
                 uint32_t steps, const std::function<void(float)>& progress = {}, HeatStepHook* hook = nullptr);

private:
    friend class HeatSession;

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

// One heat field kept on the GPU between calls, so it can be advanced a batch of steps at a time and read
// back in between (a live view). Steps continue from where the last advance() stopped, and the hook sees
// one sequential step index across all of them. Same validation and starting field as HeatSolver::run.
// The solver and the hook must outlive the session; only one thread may use it (or the solver) at a time.
class HeatSession{
public:
    HeatSession(const HeatSolver& solver, const HeatParams& params, const std::vector<uint32_t>& cells,
                HeatStepHook* hook = nullptr);
    HeatSession(const HeatSolver& solver, const HeatParams& params, const std::vector<uint32_t>& cells,
                const std::vector<float>& initial, HeatStepHook* hook = nullptr);
    HeatSession(const HeatSession&) = delete;
    HeatSession& operator=(const HeatSession&) = delete;

    // Runs `steps` more steps and waits for them.
    void advance(uint32_t steps);
    // The field after every step so far (time = stepsDone() * timeStep). Also runs the hook's readback.
    HeatGrid read();
    uint32_t stepsDone() const { return done; }

private:
    friend class HeatSolver;

    vk::raii::CommandBuffer beginCommands() const;
    void submit(vk::raii::CommandBuffer& commandBuffer) const;
    void recordUploadOnce(vk::raii::CommandBuffer& commandBuffer);
    void recordSteps(vk::raii::CommandBuffer& commandBuffer, uint32_t steps);
    void recordReadback(vk::raii::CommandBuffer& commandBuffer);
    HeatGrid download() const;

    const HeatSolver& solver;
    HeatParams params;
    std::vector<uint32_t> cells;
    HeatStepHook* hook = nullptr;
    vk::DeviceSize bytes = 0;
    vk::raii::Buffer staging = nullptr;  // initial field then cell kinds; the readback lands in the first half
    vk::raii::DeviceMemory stagingMemory = nullptr;
    vk::raii::Buffer cellBuffer = nullptr;
    vk::raii::DeviceMemory cellMemory = nullptr;
    // Ping-pong pair: step k reads fields[k % 2] and writes fields[(k + 1) % 2].
    std::array<vk::raii::Buffer, 2> fields{nullptr, nullptr};
    std::array<vk::raii::DeviceMemory, 2> fieldMemory{nullptr, nullptr};
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::array<vk::raii::DescriptorSet, 2> descriptorSets{nullptr, nullptr};
    bool uploaded = false;
    uint32_t done = 0;
};
