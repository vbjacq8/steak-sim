#include "voxelizer.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

constexpr uint32_t GROUP_SIZE = 4;  // numthreads per axis in voxelize.slang

// Must match VoxelizeParams in shaders/compute/voxelize.slang (std430 push constant layout).
struct VoxelizeParams{
    glm::vec3 origin;
    float voxelSize;
    glm::uvec3 dims;
    uint32_t triangleCount;
    uint32_t vertexStride;
    uint32_t captureUv;
};
static_assert(sizeof(VoxelizeParams) == 40);

std::vector<char> readFile(const std::string& filename){
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open " + filename);
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
}

}  // namespace

VoxelGrid makeVoxelGridLayout(glm::vec3 low, glm::vec3 high, float voxelSize){
    const glm::ivec3 lowIdx = glm::ivec3(glm::floor(low / voxelSize)) - 1;
    const glm::ivec3 highIdx = glm::ivec3(glm::ceil(high / voxelSize)) + 1;
    VoxelGrid grid;
    grid.voxelSize = voxelSize;
    grid.origin = glm::vec3(lowIdx) * voxelSize;
    grid.dims = glm::uvec3(highIdx - lowIdx);
    return grid;
}

Voxelizer::Voxelizer(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
                     const vk::raii::Queue& queue, uint32_t queueFamilyIndex, const std::string& shaderPath)
    : device(device), physicalDevice(physicalDevice), queue(queue){
    commandPool = vk::raii::CommandPool(device, vk::CommandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = queueFamilyIndex});

    std::array<vk::DescriptorSetLayoutBinding, 5> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i){
        bindings[i] = {.binding = i, .descriptorType = vk::DescriptorType::eStorageBuffer,
                       .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    setLayout = vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});

    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(VoxelizeParams)};
    pipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
        .setLayoutCount = 1, .pSetLayouts = &*setLayout,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &pushConstantRange});

    const std::vector<char> code = readFile(shaderPath);
    vk::raii::ShaderModule module(device, vk::ShaderModuleCreateInfo{
        .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data())});
    pipeline = vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{
        .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "voxelizeMain"},
        .layout = pipelineLayout});

    vk::DescriptorPoolSize poolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 5};
    descriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &poolSize});
    descriptorSet = std::move(device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo{
        .descriptorPool = descriptorPool, .descriptorSetCount = 1, .pSetLayouts = &*setLayout}).front());
}

VoxelGrid Voxelizer::voxelize(vk::Buffer vertexBuffer, uint32_t vertexStrideFloats, vk::Buffer indexBuffer,
                              uint32_t triangleCount, glm::vec3 low, glm::vec3 high, float voxelSize,
                              vk::Buffer texcoords, uint32_t texcoordCount){
    VoxelGrid grid = makeVoxelGridLayout(low, high, voxelSize);
    const bool captureUv = static_cast<bool>(texcoords) && texcoordCount > 0;
    const vk::DeviceSize voxelBufferSize = grid.cellCount() * sizeof(uint32_t);
    const vk::DeviceSize uvBufferSize = grid.cellCount() * sizeof(glm::vec2);
    auto [voxelBuffer, voxelBufferMemory] = createHostBuffer(voxelBufferSize, vk::BufferUsageFlagBits::eStorageBuffer);
    // The shader always has the UV bindings. A tiny stand-in is bound when UVs are not requested,
    // and the shader does not touch it (captureUv is 0).
    auto [vertexUvBuffer, vertexUvMemory] = createHostBuffer(16, vk::BufferUsageFlagBits::eStorageBuffer);
    auto [voxelUvBuffer, voxelUvMemory] = createHostBuffer(captureUv ? uvBufferSize : 16, vk::BufferUsageFlagBits::eStorageBuffer);

    std::array<vk::DescriptorBufferInfo, 5> bufferInfos{{
        {.buffer = vertexBuffer, .offset = 0, .range = vk::WholeSize},
        {.buffer = indexBuffer, .offset = 0, .range = vk::WholeSize},
        {.buffer = voxelBuffer, .offset = 0, .range = vk::WholeSize},
        {.buffer = captureUv ? texcoords : *vertexUvBuffer, .offset = 0, .range = vk::WholeSize},
        {.buffer = voxelUvBuffer, .offset = 0, .range = vk::WholeSize},
    }};
    std::array<vk::WriteDescriptorSet, 5> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i){
        writes[i] = {.dstSet = descriptorSet, .dstBinding = i, .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &bufferInfos[i]};
    }
    device.updateDescriptorSets(writes, {});

    const VoxelizeParams params{
        .origin = grid.origin,
        .voxelSize = grid.voxelSize,
        .dims = grid.dims,
        .triangleCount = triangleCount,
        .vertexStride = vertexStrideFloats,
        .captureUv = captureUv ? 1u : 0u,
    };
    const glm::uvec3 groups = (grid.dims + GROUP_SIZE - 1u) / GROUP_SIZE;

    vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
        .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1}).front());
    commandBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayout, 0, *descriptorSet, nullptr);
    commandBuffer.pushConstants<VoxelizeParams>(*pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, params);
    commandBuffer.dispatch(groups.x, groups.y, groups.z);

    vk::MemoryBarrier2 toHost{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
    commandBuffer.end();

    queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer}, nullptr);
    queue.waitIdle();

    grid.cells.resize(grid.cellCount());
    void* data = voxelBufferMemory.mapMemory(0, voxelBufferSize);
    std::memcpy(grid.cells.data(), data, voxelBufferSize);
    voxelBufferMemory.unmapMemory();
    if (captureUv){
        grid.uv.resize(grid.cellCount());
        void* uvs = voxelUvMemory.mapMemory(0, uvBufferSize);
        std::memcpy(grid.uv.data(), uvs, uvBufferSize);
        voxelUvMemory.unmapMemory();
    }
    return grid;
}

VoxelGrid Voxelizer::voxelize(const std::vector<glm::vec3>& positions, const std::vector<uint32_t>& indices,
                              float voxelSize){
    if (positions.empty() || indices.size() < 3) throw std::invalid_argument("voxelize: empty mesh");

    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (const glm::vec3& p : positions){
        low = glm::min(low, p);
        high = glm::max(high, p);
    }

    const vk::DeviceSize positionBytes = positions.size() * sizeof(glm::vec3);
    const vk::DeviceSize indexBytes = indices.size() * sizeof(uint32_t);
    auto [positionBuffer, positionMemory] = createHostBuffer(positionBytes, vk::BufferUsageFlagBits::eStorageBuffer);
    auto [indexBuffer, indexMemory] = createHostBuffer(indexBytes, vk::BufferUsageFlagBits::eStorageBuffer);
    std::memcpy(positionMemory.mapMemory(0, positionBytes), positions.data(), positionBytes);
    positionMemory.unmapMemory();
    std::memcpy(indexMemory.mapMemory(0, indexBytes), indices.data(), indexBytes);
    indexMemory.unmapMemory();

    return voxelize(*positionBuffer, sizeof(glm::vec3) / sizeof(float), *indexBuffer,
                    static_cast<uint32_t>(indices.size() / 3), low, high, voxelSize);
}

VoxelGrid Voxelizer::voxelize(const std::vector<glm::vec3>& positions, const std::vector<glm::vec2>& texcoords,
                              const std::vector<uint32_t>& indices, float voxelSize){
    if (positions.empty() || indices.size() < 3) throw std::invalid_argument("voxelize: empty mesh");
    if (texcoords.size() != positions.size()) throw std::invalid_argument("voxelize: texcoord count != vertex count");

    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (const glm::vec3& p : positions){
        low = glm::min(low, p);
        high = glm::max(high, p);
    }

    const vk::DeviceSize positionBytes = positions.size() * sizeof(glm::vec3);
    const vk::DeviceSize uvBytes = texcoords.size() * sizeof(glm::vec2);
    const vk::DeviceSize indexBytes = indices.size() * sizeof(uint32_t);
    auto [positionBuffer, positionMemory] = createHostBuffer(positionBytes, vk::BufferUsageFlagBits::eStorageBuffer);
    auto [uvBuffer, uvMemory] = createHostBuffer(uvBytes, vk::BufferUsageFlagBits::eStorageBuffer);
    auto [indexBuffer, indexMemory] = createHostBuffer(indexBytes, vk::BufferUsageFlagBits::eStorageBuffer);
    std::memcpy(positionMemory.mapMemory(0, positionBytes), positions.data(), positionBytes);
    positionMemory.unmapMemory();
    std::memcpy(uvMemory.mapMemory(0, uvBytes), texcoords.data(), uvBytes);
    uvMemory.unmapMemory();
    std::memcpy(indexMemory.mapMemory(0, indexBytes), indices.data(), indexBytes);
    indexMemory.unmapMemory();

    return voxelize(*positionBuffer, sizeof(glm::vec3) / sizeof(float), *indexBuffer,
                    static_cast<uint32_t>(indices.size() / 3), low, high, voxelSize, *uvBuffer,
                    static_cast<uint32_t>(texcoords.size()));
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> Voxelizer::createHostBuffer(vk::DeviceSize size,
                                                                                vk::BufferUsageFlags usage) const{
    vk::raii::Buffer buffer(device, vk::BufferCreateInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive});
    const vk::MemoryRequirements requirements = buffer.getMemoryRequirements();
    const vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    const vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();

    uint32_t memoryType = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i){
        if ((requirements.memoryTypeBits & (1u << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties){
            memoryType = i;
            break;
        }
    }
    if (memoryType == std::numeric_limits<uint32_t>::max()) throw std::runtime_error("no host-visible memory type");

    vk::raii::DeviceMemory memory(device, vk::MemoryAllocateInfo{.allocationSize = requirements.size, .memoryTypeIndex = memoryType});
    buffer.bindMemory(*memory, 0);
    return {std::move(buffer), std::move(memory)};
}
