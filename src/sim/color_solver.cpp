#include "color_solver.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

struct ColorPush{
    glm::uvec3 dims;
    float dt;
};
static_assert(sizeof(ColorPush) == 16);

constexpr vk::ImageSubresourceRange kColorRange{
    .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1};

std::vector<char> readFile(const std::string& filename){
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open " + filename);
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return buffer;
}

vk::raii::ShaderModule loadModule(const vk::raii::Device& device, const std::string& path){
    const std::vector<char> code = readFile(path);
    return vk::raii::ShaderModule(device, vk::ShaderModuleCreateInfo{
        .codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t*>(code.data())});
}

}  // namespace

ColorSolver::ColorSolver(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
                         const vk::raii::Queue& queue, uint32_t queueFamilyIndex, const std::string& initShaderPath,
                         const std::string& kineticsShaderPath, const std::string& exportShaderPath)
    : device(device), physicalDevice(physicalDevice), queue(queue){
    const auto labFeatures = physicalDevice.getFormatProperties(vk::Format::eR16G16B16A16Sfloat).optimalTilingFeatures;
    if (!physicalDevice.getFeatures().shaderStorageImageExtendedFormats ||
        !(labFeatures & vk::FormatFeatureFlagBits::eStorageImage)){
        throw std::runtime_error("RGBA16F storage images are not supported");
    }

    commandPool = vk::raii::CommandPool(device, vk::CommandPoolCreateInfo{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = queueFamilyIndex});

    const vk::PushConstantRange push{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(ColorPush)};
    auto makeLayout = [&](const std::vector<vk::DescriptorSetLayoutBinding>& bindings){
        return vk::raii::DescriptorSetLayout(device, vk::DescriptorSetLayoutCreateInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});
    };
    auto makePipeline = [&](const std::string& path, const char* entry, vk::raii::DescriptorSetLayout& setLayout,
                            vk::raii::PipelineLayout& pipelineLayout){
        pipelineLayout = vk::raii::PipelineLayout(device, vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1, .pSetLayouts = &*setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &push});
        vk::raii::ShaderModule module = loadModule(device, path);
        return vk::raii::Pipeline(device, nullptr, vk::ComputePipelineCreateInfo{
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = entry},
            .layout = pipelineLayout});
    };

    const vk::ShaderStageFlags compute = vk::ShaderStageFlagBits::eCompute;
    initLayout = makeLayout({
        {0, vk::DescriptorType::eStorageBuffer, 1, compute},
        {1, vk::DescriptorType::eStorageBuffer, 1, compute},
        {2, vk::DescriptorType::eSampledImage, 1, compute},
        {3, vk::DescriptorType::eSampler, 1, compute},
        {4, vk::DescriptorType::eStorageImage, 1, compute},
        {5, vk::DescriptorType::eStorageImage, 1, compute},
    });
    kineticsLayout = makeLayout({
        {0, vk::DescriptorType::eStorageBuffer, 1, compute},
        {1, vk::DescriptorType::eStorageBuffer, 1, compute},
        {2, vk::DescriptorType::eStorageImage, 1, compute},
        {3, vk::DescriptorType::eStorageImage, 1, compute},
        {4, vk::DescriptorType::eStorageImage, 1, compute},
    });
    exportLayout = makeLayout({
        {0, vk::DescriptorType::eStorageImage, 1, compute},
        {1, vk::DescriptorType::eStorageBuffer, 1, compute},
    });
    initPipeline = makePipeline(initShaderPath, "colorInitMain", initLayout, initPipelineLayout);
    kineticsPipeline = makePipeline(kineticsShaderPath, "colorKineticsMain", kineticsLayout, kineticsPipelineLayout);
    exportPipeline = makePipeline(exportShaderPath, "colorExportMain", exportLayout, exportPipelineLayout);

    albedoSampler = vk::raii::Sampler(device, vk::SamplerCreateInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
    });
}

void ColorSolver::prepareGrid(glm::uvec3 dimsIn, const std::vector<uint32_t>& cells, const std::vector<glm::vec2>& uv,
                              const uint8_t* rgba, uint32_t width, uint32_t height, float dtSeconds){
    const size_t count = size_t(dimsIn.x) * dimsIn.y * dimsIn.z;
    if (count == 0 || cells.size() != count || uv.size() != count){
        throw std::invalid_argument("color solver: grid, cells and UVs must describe the same volume");
    }
    if (rgba == nullptr || width == 0 || height == 0) throw std::invalid_argument("color solver: empty texture");
    if (!(dtSeconds > 0.0f)) throw std::invalid_argument("color solver: timestep must be positive seconds");
    const auto& limits = physicalDevice.getProperties().limits;
    if (dimsIn.x > limits.maxImageDimension3D || dimsIn.y > limits.maxImageDimension3D ||
        dimsIn.z > limits.maxImageDimension3D || width > limits.maxImageDimension2D || height > limits.maxImageDimension2D){
        throw std::invalid_argument("color solver: grid or texture exceeds device image limits");
    }
    const auto srgb = physicalDevice.getFormatProperties(vk::Format::eR8G8B8A8Srgb).optimalTilingFeatures;
    if (!(srgb & vk::FormatFeatureFlagBits::eSampledImage) || !(srgb & vk::FormatFeatureFlagBits::eTransferDst)){
        throw std::runtime_error("sRGB sampled textures are not supported");
    }

    dims = dimsIn;
    dt = dtSeconds;
    completedSteps = 0;
    labValues.clear();
    rgbValues.clear();

    const vk::DeviceSize cellBytes = count * sizeof(uint32_t);
    const vk::DeviceSize uvBytes = count * sizeof(glm::vec2);
    const vk::DeviceSize exportBytes = count * 2 * sizeof(glm::vec4);
    const vk::DeviceSize imageBytes = vk::DeviceSize(width) * height * 4;
    const auto hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    std::tie(cellBuffer, cellMemory) = createBuffer(cellBytes, vk::BufferUsageFlagBits::eStorageBuffer, hostFlags);
    std::tie(uvBuffer, uvMemory) = createBuffer(uvBytes, vk::BufferUsageFlagBits::eStorageBuffer, hostFlags);
    std::tie(exportBuffer, exportMemory) = createBuffer(exportBytes, vk::BufferUsageFlagBits::eStorageBuffer, hostFlags);
    std::memcpy(cellMemory.mapMemory(0, cellBytes), cells.data(), cellBytes);
    cellMemory.unmapMemory();
    std::memcpy(uvMemory.mapMemory(0, uvBytes), uv.data(), uvBytes);
    uvMemory.unmapMemory();

    auto [staging, stagingMemory] = createBuffer(imageBytes, vk::BufferUsageFlagBits::eTransferSrc, hostFlags);
    std::memcpy(stagingMemory.mapMemory(0, imageBytes), rgba, imageBytes);
    stagingMemory.unmapMemory();

    albedo = createImage(vk::ImageType::e2D, vk::Format::eR8G8B8A8Srgb, {width, height, 1},
                         vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::ImageViewType::e2D);
    const vk::Extent3D volume{dims.x, dims.y, dims.z};
    const vk::ImageUsageFlags labUsage = vk::ImageUsageFlagBits::eStorage;
    lab0 = createImage(vk::ImageType::e3D, vk::Format::eR16G16B16A16Sfloat, volume, labUsage, vk::ImageViewType::e3D);
    labImage[0] = createImage(vk::ImageType::e3D, vk::Format::eR16G16B16A16Sfloat, volume, labUsage, vk::ImageViewType::e3D);
    labImage[1] = createImage(vk::ImageType::e3D, vk::Format::eR16G16B16A16Sfloat, volume, labUsage, vk::ImageViewType::e3D);

    vk::raii::CommandBuffer commandBuffer = beginCommands();
    vk::ImageMemoryBarrier2 toTransfer{
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = albedo.image,
        .subresourceRange = kColorRange};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toTransfer});
    vk::BufferImageCopy region{
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .imageExtent = {width, height, 1}};
    commandBuffer.copyBufferToImage(*staging, *albedo.image, vk::ImageLayout::eTransferDstOptimal, region);
    vk::ImageMemoryBarrier2 toSample{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = albedo.image,
        .subresourceRange = kColorRange};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toSample});
    submit(commandBuffer);

    std::array<vk::DescriptorPoolSize, 4> poolSizes{{
        {.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 12},
        {.type = vk::DescriptorType::eSampledImage, .descriptorCount = 2},
        {.type = vk::DescriptorType::eSampler, .descriptorCount = 2},
        {.type = vk::DescriptorType::eStorageImage, .descriptorCount = 16},
    }};
    descriptorPool = vk::raii::DescriptorPool(device, vk::DescriptorPoolCreateInfo{
        .maxSets = 8, .poolSizeCount = static_cast<uint32_t>(poolSizes.size()), .pPoolSizes = poolSizes.data()});
    const std::array<vk::DescriptorSetLayout, 5> layouts{
        *initLayout, *exportLayout, *exportLayout, *kineticsLayout, *kineticsLayout};
    vk::raii::DescriptorSets allocated(device, vk::DescriptorSetAllocateInfo{
        .descriptorPool = descriptorPool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()});
    initSet = std::move(allocated[0]);
    exportSet[0] = std::move(allocated[1]);
    exportSet[1] = std::move(allocated[2]);
    kineticsSet[0] = std::move(allocated[3]);
    kineticsSet[1] = std::move(allocated[4]);

    vk::DescriptorBufferInfo cellsInfo{.buffer = cellBuffer, .range = vk::WholeSize};
    vk::DescriptorBufferInfo uvInfo{.buffer = uvBuffer, .range = vk::WholeSize};
    vk::DescriptorBufferInfo exportInfo{.buffer = exportBuffer, .range = vk::WholeSize};
    vk::DescriptorImageInfo albedoInfo{.imageView = albedo.view, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo samplerInfo{.sampler = albedoSampler};
    vk::DescriptorImageInfo lab0Info{.imageView = lab0.view, .imageLayout = vk::ImageLayout::eGeneral};
    vk::DescriptorImageInfo labInfo[2]{{.imageView = labImage[0].view, .imageLayout = vk::ImageLayout::eGeneral},
                                       {.imageView = labImage[1].view, .imageLayout = vk::ImageLayout::eGeneral}};

    std::array<vk::WriteDescriptorSet, 6> initWrites{{
        {.dstSet = initSet, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &cellsInfo},
        {.dstSet = initSet, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &uvInfo},
        {.dstSet = initSet, .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &albedoInfo},
        {.dstSet = initSet, .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler,
         .pImageInfo = &samplerInfo},
        {.dstSet = initSet, .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &lab0Info},
        {.dstSet = initSet, .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &labInfo[0]},
    }};
    std::array<vk::WriteDescriptorSet, 2> exportWrites{{
        {.dstSet = exportSet[0], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &labInfo[0]},
        {.dstSet = exportSet[0], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &exportInfo},
    }};
    // The second export set is written below; both share the same buffer.
    std::array<vk::WriteDescriptorSet, 2> exportWrites1{{
        {.dstSet = exportSet[1], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &labInfo[1]},
        {.dstSet = exportSet[1], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer,
         .pBufferInfo = &exportInfo},
    }};
    device.updateDescriptorSets(initWrites, {});
    device.updateDescriptorSets(exportWrites, {});
    device.updateDescriptorSets(exportWrites1, {});
}

void ColorSolver::prepare(vk::Buffer temperature0, vk::Buffer temperature1){
    if (!*kineticsSet[0]) throw std::logic_error("color solver: prepareGrid before the heat run");
    // After heat step s the new temperature is in buffer (s + 1) % 2, and the color step reads labImage[s % 2].
    const vk::Buffer temps[2] = {temperature1, temperature0};
    const int labIn[2] = {0, 1};
    const int labOut[2] = {1, 0};
    vk::DescriptorBufferInfo tempInfo[2]{{.buffer = temps[0], .range = vk::WholeSize},
                                         {.buffer = temps[1], .range = vk::WholeSize}};
    vk::DescriptorBufferInfo cellsInfo{.buffer = cellBuffer, .range = vk::WholeSize};
    vk::DescriptorImageInfo lab0Info{.imageView = lab0.view, .imageLayout = vk::ImageLayout::eGeneral};
    vk::DescriptorImageInfo labInfo[2]{{.imageView = labImage[0].view, .imageLayout = vk::ImageLayout::eGeneral},
                                       {.imageView = labImage[1].view, .imageLayout = vk::ImageLayout::eGeneral}};
    std::array<vk::WriteDescriptorSet, 10> writes{};
    for (int set = 0; set < 2; ++set){
        const int base = set * 5;
        writes[base + 0] = {.dstSet = kineticsSet[set], .dstBinding = 0, .descriptorCount = 1,
                            .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &tempInfo[set]};
        writes[base + 1] = {.dstSet = kineticsSet[set], .dstBinding = 1, .descriptorCount = 1,
                            .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &cellsInfo};
        writes[base + 2] = {.dstSet = kineticsSet[set], .dstBinding = 2, .descriptorCount = 1,
                            .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &labInfo[labIn[set]]};
        writes[base + 3] = {.dstSet = kineticsSet[set], .dstBinding = 3, .descriptorCount = 1,
                            .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &lab0Info};
        writes[base + 4] = {.dstSet = kineticsSet[set], .dstBinding = 4, .descriptorCount = 1,
                            .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &labInfo[labOut[set]]};
    }
    device.updateDescriptorSets(writes, {});
}

void ColorSolver::recordInit(vk::raii::CommandBuffer& commandBuffer){
    completedSteps = 0;
    const vk::Image images[3] = {*lab0.image, *labImage[0].image, *labImage[1].image};
    std::array<vk::ImageMemoryBarrier2, 3> toGeneral{};
    for (int i = 0; i < 3; ++i){
        toGeneral[i] = {.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
                        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                        .oldLayout = vk::ImageLayout::eUndefined,
                        .newLayout = vk::ImageLayout::eGeneral,
                        .image = images[i],
                        .subresourceRange = kColorRange};
    }
    vk::MemoryBarrier2 hostToShader{.srcStageMask = vk::PipelineStageFlagBits2::eHost,
                                    .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
                                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderSampledRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &hostToShader,
                                                      .imageMemoryBarrierCount = static_cast<uint32_t>(toGeneral.size()),
                                                      .pImageMemoryBarriers = toGeneral.data()});
    dispatch(commandBuffer, *initPipeline, *initPipelineLayout, *initSet);
    vk::MemoryBarrier2 written{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                               .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                               .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                               .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &written});
}

void ColorSolver::recordAfterStep(vk::raii::CommandBuffer& commandBuffer, uint32_t stepIndex){
    if (stepIndex != completedSteps) throw std::logic_error("color solver: heat steps were not sequential");
    dispatch(commandBuffer, *kineticsPipeline, *kineticsPipelineLayout, *kineticsSet[stepIndex % 2]);
    ++completedSteps;
    vk::MemoryBarrier2 written{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                               .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                               .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                               .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &written});
}

void ColorSolver::recordReadback(vk::raii::CommandBuffer& commandBuffer){
    dispatch(commandBuffer, *exportPipeline, *exportPipelineLayout, *exportSet[completedSteps % 2]);
    vk::MemoryBarrier2 toHost{.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                              .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                              .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                              .dstAccessMask = vk::AccessFlagBits2::eHostRead};
    commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toHost});
}

void ColorSolver::download(){
    const size_t count = size_t(dims.x) * dims.y * dims.z;
    const vk::DeviceSize bytes = count * 2 * sizeof(glm::vec4);
    auto* data = static_cast<const glm::vec4*>(exportMemory.mapMemory(0, bytes));
    labValues.resize(count);
    rgbValues.resize(count);
    for (size_t i = 0; i < count; ++i){
        labValues[i] = data[i];
        rgbValues[i] = glm::vec3(data[count + i]);
    }
    exportMemory.unmapMemory();
}

void ColorSolver::runExplicitTemperatures(const std::vector<std::vector<float>>& temperatureSteps){
    const size_t count = size_t(dims.x) * dims.y * dims.z;
    if (count == 0) throw std::logic_error("color solver: prepareGrid before runExplicitTemperatures");
    for (const std::vector<float>& field : temperatureSteps){
        if (field.size() != count) throw std::invalid_argument("color solver: temperature field size != grid size");
    }
    const vk::DeviceSize bytes = count * sizeof(float);
    auto [temperature, memory] = createBuffer(bytes, vk::BufferUsageFlagBits::eStorageBuffer,
                                              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    auto* mapped = static_cast<float*>(memory.mapMemory(0, bytes));
    prepare(*temperature, *temperature);

    vk::raii::CommandBuffer init = beginCommands();
    recordInit(init);
    submit(init);
    vk::MemoryBarrier2 hostToShader{.srcStageMask = vk::PipelineStageFlagBits2::eHost,
                                    .srcAccessMask = vk::AccessFlagBits2::eHostWrite,
                                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead};
    for (uint32_t step = 0; step < temperatureSteps.size(); ++step){
        std::memcpy(mapped, temperatureSteps[step].data(), bytes);
        vk::raii::CommandBuffer commandBuffer = beginCommands();
        commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &hostToShader});
        recordAfterStep(commandBuffer, step);
        submit(commandBuffer);
    }
    memory.unmapMemory();
    vk::raii::CommandBuffer readback = beginCommands();
    recordReadback(readback);
    submit(readback);
    download();
}

void ColorSolver::dispatch(vk::raii::CommandBuffer& commandBuffer, vk::Pipeline pipeline, vk::PipelineLayout layout,
                           vk::DescriptorSet set) const{
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, set, nullptr);
    commandBuffer.pushConstants<ColorPush>(layout, vk::ShaderStageFlagBits::eCompute, 0, ColorPush{.dims = dims, .dt = dt});
    const glm::uvec3 groups = (dims + glm::uvec3(3u)) / glm::uvec3(4u);
    commandBuffer.dispatch(groups.x, groups.y, groups.z);
}

vk::raii::CommandBuffer ColorSolver::beginCommands() const{
    vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, vk::CommandBufferAllocateInfo{
        .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1}).front());
    commandBuffer.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return commandBuffer;
}

void ColorSolver::submit(vk::raii::CommandBuffer& commandBuffer) const{
    commandBuffer.end();
    queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer}, nullptr);
    queue.waitIdle();
}

ColorSolver::GpuImage ColorSolver::createImage(vk::ImageType type, vk::Format format, vk::Extent3D extent,
                                               vk::ImageUsageFlags usage, vk::ImageViewType viewType) const{
    GpuImage result;
    result.image = vk::raii::Image(device, vk::ImageCreateInfo{
        .imageType = type,
        .format = format,
        .extent = extent,
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive});
    const vk::MemoryRequirements requirements = result.image.getMemoryRequirements();
    const vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
    uint32_t memoryType = std::numeric_limits<uint32_t>::max();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i){
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)){
            memoryType = i;
            break;
        }
    }
    if (memoryType == std::numeric_limits<uint32_t>::max()){
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i){
            if (requirements.memoryTypeBits & (1u << i)){
                memoryType = i;
                break;
            }
        }
    }
    if (memoryType == std::numeric_limits<uint32_t>::max()) throw std::runtime_error("no image memory type");
    result.memory = vk::raii::DeviceMemory(device, vk::MemoryAllocateInfo{
        .allocationSize = requirements.size, .memoryTypeIndex = memoryType});
    result.image.bindMemory(*result.memory, 0);
    result.view = vk::raii::ImageView(device, vk::ImageViewCreateInfo{
        .image = result.image, .viewType = viewType, .format = format, .subresourceRange = kColorRange});
    return result;
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> ColorSolver::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
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
