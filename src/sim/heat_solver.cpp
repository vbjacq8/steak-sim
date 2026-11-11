#include "heat_solver.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>

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

}  // namespace

std::vector<uint32_t> makePanCells(glm::uvec3 dims, const std::vector<uint32_t>& solid){
    const size_t slab = size_t(dims.x) * dims.y;
    if (dims.z == 0 || solid.size() != slab * dims.z) throw std::invalid_argument("makePanCells: solid mask size != grid size");
    std::vector<uint32_t> cells(solid.size());
    for (size_t i = 0; i < cells.size(); ++i){
        const bool panLayer = i < slab;  // z == 0
        cells[i] = panLayer ? CELL_PAN : (solid[i] != 0 ? CELL_SOLID : CELL_AIR);
    }
    return cells;
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
    std::vector<float> initial(cells.size(), params.initialTemp);
    for (size_t i = 0; i < cells.size(); ++i){
        if (cells[i] == CELL_PAN) initial[i] = params.panTemp;
        else if (cells[i] == CELL_AIR) initial[i] = params.airTemp;
    }
    return run(params, cells, initial, steps, progress, hook);
}

HeatGrid HeatSolver::run(const HeatParams& params, const std::vector<uint32_t>& cells, const std::vector<float>& initial,
                         uint32_t steps, const std::function<void(float)>& progress, HeatStepHook* hook){
    HeatGrid grid{.dims = params.dims, .spacing = params.spacing, .time = steps * params.timeStep, .cells = cells};
    if (grid.cellCount() == 0) throw std::invalid_argument("heat solver: empty grid");
    if (initial.size() != grid.cellCount()) throw std::invalid_argument("heat solver: initial field size != grid size");
    if (cells.size() != grid.cellCount()) throw std::invalid_argument("heat solver: cell kinds size != grid size");
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

    // Staging holds the initial field then the cell kinds; the result is read back into the first half.
    static_assert(sizeof(float) == sizeof(uint32_t));
    const vk::DeviceSize bytes = grid.cellCount() * sizeof(float);
    auto [staging, stagingMemory] = createBuffer(2 * bytes,
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto* mapped = static_cast<char*>(stagingMemory.mapMemory(0, 2 * bytes));
    std::memcpy(mapped, initial.data(), bytes);
    std::memcpy(mapped + bytes, cells.data(), bytes);
    stagingMemory.unmapMemory();
    auto [cellBuffer, cellMemory] = createBuffer(bytes,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Ping-pong pair: step k reads fields[k % 2] and writes fields[(k + 1) % 2].
    const vk::BufferUsageFlags fieldUsage = vk::BufferUsageFlagBits::eStorageBuffer |
                                            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    std::array<std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>, 2> fields{
        createBuffer(bytes, fieldUsage, vk::MemoryPropertyFlagBits::eDeviceLocal),
        createBuffer(bytes, fieldUsage, vk::MemoryPropertyFlagBits::eDeviceLocal)};

    vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 6};
    vk::raii::DescriptorPool descriptorPool(device, vk::DescriptorPoolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &poolSize});
    const std::array<vk::DescriptorSetLayout, 2> layouts{*setLayout, *setLayout};
    vk::raii::DescriptorSets descriptorSets(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = descriptorPool, .descriptorSetCount = 2, .pSetLayouts = layouts.data()});
    for (uint32_t k = 0; k < 2; ++k){
        std::array<vk::DescriptorBufferInfo, 3> bufferInfos{{
            {.buffer = fields[k].first, .offset = 0, .range = vk::WholeSize},
            {.buffer = fields[1 - k].first, .offset = 0, .range = vk::WholeSize},
            {.buffer = cellBuffer, .offset = 0, .range = vk::WholeSize},
        }};
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (uint32_t b = 0; b < writes.size(); ++b){
            writes[b] = {.dstSet = descriptorSets[k], .dstBinding = b, .descriptorCount = 1,
                         .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &bufferInfos[b]};
        }
        device.updateDescriptorSets(writes, {});
    }
    if (hook) hook->prepare(*fields[0].first, *fields[1].first);

    const HeatStepParams stepParams{.dims = params.dims, .fourier = params.fourier(), .panTemp = params.panTemp,
                                    .panCoupling = params.panCoupling(), .airTemp = params.airTemp,
                                    .airCoupling = params.airCoupling()};
    const glm::uvec3 groups = (params.dims + GROUP_SIZE - 1u) / GROUP_SIZE;
    const vk::BufferCopy wholeBuffer{.size = bytes};
    vk::MemoryBarrier2 uploadToCompute{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead};
    vk::MemoryBarrier2 betweenSteps{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite |
                         vk::AccessFlagBits2::eTransferRead};
    vk::MemoryBarrier2 toHost{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead};

    auto beginCommands = [&](){
        vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
            .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1}).front());
        commandBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return commandBuffer;
    };
    auto submit = [&](vk::raii::CommandBuffer& commandBuffer){
        commandBuffer.end();
        queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer}, nullptr);
        queue.waitIdle();
    };
    auto recordUpload = [&](vk::raii::CommandBuffer& commandBuffer){
        commandBuffer.copyBuffer(*staging, *fields[0].first, wholeBuffer);
        commandBuffer.copyBuffer(*staging, *cellBuffer, vk::BufferCopy{.srcOffset = bytes, .size = bytes});
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &uploadToCompute});
        if (hook) hook->recordInit(commandBuffer);
    };
    auto recordDispatches = [&](vk::raii::CommandBuffer& commandBuffer, uint32_t begin, uint32_t end){
        for (uint32_t step = begin; step < end; ++step){
            // Rebound every step: a color hook binds its own pipeline and overwrites push constants.
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
            commandBuffer.pushConstants<HeatStepParams>(*pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, stepParams);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, *descriptorSets[step % 2], nullptr);
            commandBuffer.dispatch(groups.x, groups.y, groups.z);
            commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &betweenSteps});
            if (hook) hook->recordAfterStep(commandBuffer, step);
        }
    };
    auto recordReadback = [&](vk::raii::CommandBuffer& commandBuffer){
        if (hook) hook->recordReadback(commandBuffer);
        commandBuffer.copyBuffer(*fields[steps % 2].first, *staging, wholeBuffer);
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
    };

    // One submit keeps the steady-state path to a single queue wait. A progress callback submits in batches so the
    // fraction can advance while the steps are actually running.
    if (!progress || steps == 0){
        vk::raii::CommandBuffer commandBuffer = beginCommands();
        recordUpload(commandBuffer);
        recordDispatches(commandBuffer, 0, steps);
        recordReadback(commandBuffer);
        submit(commandBuffer);
    } else{
        {
            vk::raii::CommandBuffer commandBuffer = beginCommands();
            recordUpload(commandBuffer);
            submit(commandBuffer);
        }
        const uint32_t batches = std::min(steps, 32u);
        const uint32_t batchSteps = (steps + batches - 1u) / batches;
        for (uint32_t begin = 0; begin < steps; begin += batchSteps){
            const uint32_t end = std::min(steps, begin + batchSteps);
            vk::raii::CommandBuffer commandBuffer = beginCommands();
            recordDispatches(commandBuffer, begin, end);
            submit(commandBuffer);
            progress(static_cast<float>(end) / static_cast<float>(steps));
        }
        vk::raii::CommandBuffer commandBuffer = beginCommands();
        recordReadback(commandBuffer);
        submit(commandBuffer);
    }

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
