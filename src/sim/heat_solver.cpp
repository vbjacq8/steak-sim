#include "heat_solver.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace {

const glm::uvec3 GROUP_SIZE(32, 4, 2);  // numthreads in finitedifference.slang

// Must match HeatStepParams in shaders/compute/finitedifference.slang (std430 push constant layout).
struct HeatStepParams{
    glm::uvec3 dims;
    float fourier;
    float panTemp;
    float panCoupling;
    float airTemp;
    float airCoupling;
};
static_assert(sizeof(HeatStepParams) == 32);

std::vector<char> readFile(const std::string& filename){
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open " + filename);
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
}

std::vector<float> startingField(const HeatParams& params, const std::vector<uint32_t>& cells){
    std::vector<float> initial(cells.size(), params.initialTemp);
    for (size_t i = 0; i < cells.size(); ++i){
        if (cells[i] == CELL_PAN) initial[i] = params.panTemp;
        else if (cells[i] == CELL_AIR) initial[i] = params.airTemp;
    }
    return initial;
}

}  // namespace

std::vector<uint32_t> makePanCells(glm::uvec3 dims, const std::vector<uint32_t>& solid, uint32_t panAxis){
    const size_t count = size_t(dims.x) * dims.y * dims.z;
    if (count == 0 || solid.size() != count) throw std::invalid_argument("makePanCells: solid mask size != grid size");
    if (panAxis > 2) throw std::invalid_argument("makePanCells: pan axis must be 0, 1 or 2");
    std::vector<uint32_t> cells(solid.size());
    for (uint32_t z = 0; z < dims.z; ++z){
        for (uint32_t y = 0; y < dims.y; ++y){
            for (uint32_t x = 0; x < dims.x; ++x){
                const size_t i = x + dims.x * (y + size_t(dims.y) * z);
                const bool panLayer = glm::uvec3(x, y, z)[panAxis] == 0;
                cells[i] = panLayer ? CELL_PAN : (solid[i] != 0 ? CELL_SOLID : CELL_AIR);
            }
        }
    }
    return cells;
}

std::vector<uint32_t> makeGrillCells(glm::uvec3 dims, const std::vector<uint32_t>& solid, uint32_t panAxis, uint32_t spokes,
                                     uint32_t spokeCells){
    if (spokes == 0 || spokeCells == 0) throw std::invalid_argument("makeGrillCells: need at least one spoke one cell wide");
    std::vector<uint32_t> cells = makePanCells(dims, solid, panAxis);
    const uint32_t u = (panAxis + 1) % 3;
    const uint32_t v = (panAxis + 2) % 3;
    const uint32_t across = dims[u] >= dims[v] ? u : v;
    std::vector<bool> onSpoke(dims[across], false);
    for (uint32_t i = 0; i < spokes; ++i){
        const double center = (i + 0.5) * dims[across] / spokes;
        const long start = std::lround(center - 0.5 * spokeCells);
        for (long c = std::max(0L, start); c < std::min<long>(start + spokeCells, dims[across]); ++c) onSpoke[c] = true;
    }
    for (uint32_t z = 0; z < dims.z; ++z){
        for (uint32_t y = 0; y < dims.y; ++y){
            for (uint32_t x = 0; x < dims.x; ++x){
                const glm::uvec3 cell(x, y, z);
                if (cell[panAxis] != 0 || onSpoke[cell[across]]) continue;
                cells[x + dims.x * (y + size_t(dims.y) * z)] = CELL_AIR;
            }
        }
    }
    return cells;
}

std::vector<uint32_t> makeCookCells(const HeatParams& params, const std::vector<uint32_t>& solid){
    if (params.grillSpokes == 0) return makePanCells(params.dims, solid, params.panAxis);
    if (!(params.spokeWidth > 0.0f) || !(params.spacing > 0.0f)) throw std::invalid_argument("makeCookCells: spoke width must be positive");
    const auto spokeCells = static_cast<uint32_t>(std::max(1L, std::lround(params.spokeWidth / params.spacing)));
    return makeGrillCells(params.dims, solid, params.panAxis, params.grillSpokes, spokeCells);
}

std::vector<uint32_t> makeBlockCells(glm::uvec3 dims){
    return makePanCells(dims, std::vector<uint32_t>(size_t(dims.x) * dims.y * dims.z, 1u));
}

HeatSolver::HeatSolver(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
                       const vk::raii::Queue& queue, uint32_t queueFamilyIndex, const std::string& shaderPath)
    : device(device), physicalDevice(physicalDevice), queue(queue){
    commandPool = vk::raii::CommandPool(device, vk::CommandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = queueFamilyIndex});

    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i){
        bindings[i] = {.binding = i, .descriptorType = vk::DescriptorType::eStorageBuffer,
                       .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    setLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});

    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(HeatStepParams)};
    pipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1, .pSetLayouts = &*setLayout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange});

    const std::vector<char> code = readFile(shaderPath);
    vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
        .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data())});
    pipeline = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{
        .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "heatStepMain"},
        .layout = pipelineLayout});
}

HeatGrid HeatSolver::run(const HeatParams& params, uint32_t steps, const std::function<void(float)>& progress,
                         HeatStepHook* hook){
    return run(params, makeBlockCells(params.dims), steps, progress, hook);
}

HeatGrid HeatSolver::run(const HeatParams& params, const std::vector<uint32_t>& cells, uint32_t steps,
                         const std::function<void(float)>& progress, HeatStepHook* hook){
    return run(params, cells, startingField(params, cells), steps, progress, hook);
}

HeatGrid HeatSolver::run(const HeatParams& params, const std::vector<uint32_t>& cells, const std::vector<float>& initial,
                         uint32_t steps, const std::function<void(float)>& progress, HeatStepHook* hook){
    HeatSession session(*this, params, cells, initial, hook);

    // One submit keeps the steady-state path to a single queue wait. A progress callback submits in batches so the
    // fraction can advance while the steps are actually running.
    if (!progress || steps == 0){
        vk::raii::CommandBuffer commandBuffer = session.beginCommands();
        session.recordUploadOnce(commandBuffer);
        session.recordSteps(commandBuffer, steps);
        session.recordReadback(commandBuffer);
        session.submit(commandBuffer);
        return session.download();
    }
    const uint32_t batches = std::min(steps, 32u);
    const uint32_t batchSteps = (steps + batches - 1u) / batches;
    for (uint32_t begin = 0; begin < steps; begin += batchSteps){
        const uint32_t end = std::min(steps, begin + batchSteps);
        session.advance(end - begin);
        progress(static_cast<float>(end) / static_cast<float>(steps));
    }
    return session.read();
}

HeatSession::HeatSession(const HeatSolver& solver, const HeatParams& params, const std::vector<uint32_t>& cells,
                         HeatStepHook* hook)
    : HeatSession(solver, params, cells, startingField(params, cells), hook){}

HeatSession::HeatSession(const HeatSolver& solverIn, const HeatParams& paramsIn, const std::vector<uint32_t>& cellsIn,
                         const std::vector<float>& initial, HeatStepHook* hookIn)
    : solver(solverIn), params(paramsIn), cells(cellsIn), hook(hookIn){
    const size_t count = size_t(params.dims.x) * params.dims.y * params.dims.z;
    if (count == 0) throw std::invalid_argument("heat solver: empty grid");
    if (initial.size() != count) throw std::invalid_argument("heat solver: initial field size != grid size");
    if (cells.size() != count) throw std::invalid_argument("heat solver: cell kinds size != grid size");
    if (!(params.spacing > 0.0f) || !(params.timeStep > 0.0f) || !(params.diffusivity > 0.0f) ||
        !(params.conductivity > 0.0f) || !(params.panTransfer >= 0.0f) || !(params.airTransfer >= 0.0f)){
        throw std::invalid_argument("heat solver: spacing, time step, diffusivity and conductivity must be positive, "
                                    "pan and air heat transfer non-negative");
    }
    if (params.fourier() > params.maxStableFourier()){
        const float maxStep = params.maxStableFourier() * params.spacing * params.spacing / params.diffusivity;
        throw std::invalid_argument("heat solver: unstable, alpha*dt/dx^2 = " + std::to_string(params.fourier()) +
                                    " > " + std::to_string(params.maxStableFourier()) + "; use dt <= " +
                                    std::to_string(maxStep) + " s for dx = " + std::to_string(params.spacing) + " m");
    }

    static_assert(sizeof(float) == sizeof(uint32_t));
    bytes = count * sizeof(float);
    std::tie(staging, stagingMemory) = solver.createBuffer(2 * bytes,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto* mapped = static_cast<char*>(stagingMemory.mapMemory(0, 2 * bytes));
    std::memcpy(mapped, initial.data(), bytes);
    std::memcpy(mapped + bytes, cells.data(), bytes);
    stagingMemory.unmapMemory();
    std::tie(cellBuffer, cellMemory) = solver.createBuffer(bytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    const vk::BufferUsageFlags fieldUsage = vk::BufferUsageFlagBits::eStorageBuffer |
                                            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    for (int k = 0; k < 2; ++k){
        std::tie(fields[k], fieldMemory[k]) = solver.createBuffer(bytes, fieldUsage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 6};
    descriptorPool = vk::raii::DescriptorPool(solver.device, vk::DescriptorPoolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &poolSize});
    const std::array<vk::DescriptorSetLayout, 2> layouts{*solver.setLayout, *solver.setLayout};
    vk::raii::DescriptorSets allocated(solver.device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = descriptorPool, .descriptorSetCount = 2, .pSetLayouts = layouts.data()});
    for (uint32_t k = 0; k < 2; ++k){
        descriptorSets[k] = std::move(allocated[k]);
        std::array<vk::DescriptorBufferInfo, 3> bufferInfos{{
            {.buffer = fields[k], .offset = 0, .range = vk::WholeSize},
            {.buffer = fields[1 - k], .offset = 0, .range = vk::WholeSize},
            {.buffer = cellBuffer, .offset = 0, .range = vk::WholeSize},
        }};
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (uint32_t b = 0; b < writes.size(); ++b){
            writes[b] = {.dstSet = descriptorSets[k], .dstBinding = b, .descriptorCount = 1,
                         .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &bufferInfos[b]};
        }
        solver.device.updateDescriptorSets(writes, {});
    }
    if (hook) hook->prepare(*fields[0], *fields[1]);
}

void HeatSession::advance(uint32_t steps){
    vk::raii::CommandBuffer commandBuffer = beginCommands();
    recordUploadOnce(commandBuffer);
    recordSteps(commandBuffer, steps);
    submit(commandBuffer);
}

HeatGrid HeatSession::read(){
    vk::raii::CommandBuffer commandBuffer = beginCommands();
    recordUploadOnce(commandBuffer);
    recordReadback(commandBuffer);
    submit(commandBuffer);
    return download();
}

vk::raii::CommandBuffer HeatSession::beginCommands() const{
    vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(solver.device, vk::CommandBufferAllocateInfo{
        .commandPool = solver.commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1}).front());
    commandBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return commandBuffer;
}

void HeatSession::submit(vk::raii::CommandBuffer& commandBuffer) const{
    commandBuffer.end();
    solver.queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer}, nullptr);
    solver.queue.waitIdle();
}

void HeatSession::recordUploadOnce(vk::raii::CommandBuffer& commandBuffer){
    if (uploaded) return;
    uploaded = true;
    commandBuffer.copyBuffer(*staging, *fields[0], vk::BufferCopy{.size = bytes});
    commandBuffer.copyBuffer(*staging, *cellBuffer, vk::BufferCopy{.srcOffset = bytes, .size = bytes});
    const vk::MemoryBarrier2 uploadToCompute{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &uploadToCompute});
    if (hook) hook->recordInit(commandBuffer);
}

void HeatSession::recordSteps(vk::raii::CommandBuffer& commandBuffer, uint32_t steps){
    // A readback between batches copies out of buffers the next steps overwrite.
    const vk::MemoryBarrier2 afterReadback{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
    if (done > 0) commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &afterReadback});
    const vk::MemoryBarrier2 betweenSteps{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite |
                         vk::AccessFlagBits2::eTransferRead};
    const HeatStepParams stepParams{.dims = params.dims, .fourier = params.fourier(), .panTemp = params.panTemp,
                                    .panCoupling = params.panCoupling(), .airTemp = params.airTemp,
                                    .airCoupling = params.airCoupling()};
    const glm::uvec3 groups = (params.dims + GROUP_SIZE - 1u) / GROUP_SIZE;
    for (uint32_t end = done + steps; done < end; ++done){
        // Rebound every step: a color hook binds its own pipeline and overwrites push constants.
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *solver.pipeline);
        commandBuffer.pushConstants<HeatStepParams>(*solver.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, stepParams);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *solver.pipelineLayout, 0, *descriptorSets[done % 2], nullptr);
        commandBuffer.dispatch(groups.x, groups.y, groups.z);
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &betweenSteps});
        if (hook) hook->recordAfterStep(commandBuffer, done);
    }
}

void HeatSession::recordReadback(vk::raii::CommandBuffer& commandBuffer){
    if (hook) hook->recordReadback(commandBuffer);
    commandBuffer.copyBuffer(*fields[done % 2], *staging, vk::BufferCopy{.size = bytes});
    const vk::MemoryBarrier2 toHost{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
}

HeatGrid HeatSession::download() const{
    HeatGrid grid{.dims = params.dims, .spacing = params.spacing, .time = done * params.timeStep, .panAxis = params.panAxis,
                  .cells = cells};
    grid.temperature.resize(grid.cellCount());
    std::memcpy(grid.temperature.data(), stagingMemory.mapMemory(0, bytes), bytes);
    stagingMemory.unmapMemory();
    return grid;
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> HeatSolver::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                                           vk::MemoryPropertyFlags properties) const{
    vk::raii::Buffer buffer(device, vk::BufferCreateInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive});
    const vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
    const vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

    uint32_t memoryType = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i){
        if ((requirements.memoryTypeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties){
            memoryType = i;
            break;
        }
    }
    if (memoryType == std::numeric_limits<uint32_t>::max()) throw std::runtime_error("no suitable memory type");

    vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{.allocationSize = requirements.size, .memoryTypeIndex = memoryType});
    buffer.bindMemory(*memory, 0);
    return {std::move(buffer), std::move(memory)};
}
