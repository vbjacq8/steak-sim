#pragma once

#include "heat_solver.hpp"

#include <cstdint>
#include <string>
#include <vector>

// Temperature-driven CIELAB color of the voxel grid.
//
// Lab is stored in RGBA16F 3D images: R = L*, G = a*, B = b*, A = 1 for CELL_SOLID and 0 otherwise.
// An immutable copy keeps the raw Lab. Each heat step then raises every solid cell's peak temperature
// (Celsius, an R32F 3D image) and writes the doneness color for that peak (color_model.hpp).
//
// Per-voxel UVs come from Voxelizer. Occupied voxels, including the interior, use the barycentric
// UV of the closest point on the mesh. Interior voxels therefore inherit the nearest surface color
// (the crust texture bleeds inward). The voxelizer does not retain a triangle id, and a voxel
// position is not a texture coordinate, so this is the correspondence the existing triangle loop
// can produce without a second mesh-query system.
//
// The heat field is degrees Fahrenheit. `dt` is HeatParams::timeStep in seconds. The Open3D viewer
// cannot sample these images, so color_export.slang converts the current Lab image to linear sRGB
// and the window draws that. The device, physical device and queue must outlive the solver.
class ColorSolver : public HeatStepHook{
public:
    ColorSolver(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
                const vk::raii::Queue& queue, uint32_t queueFamilyIndex, const std::string& initShaderPath,
                const std::string& kineticsShaderPath, const std::string& exportShaderPath);

    // `uv` is one texture coordinate per cell (empty cells may be anything; they are not sampled).
    // `rgba` is tightly packed sRGB8. `dtSeconds` is the heat timestep.
    void prepareGrid(glm::uvec3 dims, const std::vector<uint32_t>& cells, const std::vector<glm::vec2>& uv,
                     const uint8_t* rgba, uint32_t width, uint32_t height, float dtSeconds);

    void prepare(vk::Buffer temperature0, vk::Buffer temperature1) override;
    void recordInit(vk::raii::CommandBuffer& commandBuffer) override;
    void recordAfterStep(vk::raii::CommandBuffer& commandBuffer, uint32_t stepIndex) override;
    void recordReadback(vk::raii::CommandBuffer& commandBuffer) override;

    // Maps the export buffer. The queue that ran recordReadback must be idle.
    void download();

    // Initializes from the texture, then runs one color step per temperature field.
    // Each field is degrees Fahrenheit, one value per cell. Submits its own work and downloads.
    void runExplicitTemperatures(const std::vector<std::vector<float>>& temperatureSteps);

    const std::vector<glm::vec4>& lab() const { return labValues; }
    const std::vector<glm::vec3>& linearRgb() const { return rgbValues; }
    const std::vector<glm::vec3>& initialLinearRgb() const { return initialRgbValues; }
    // Hottest temperature each cell reached, Celsius; kNoPeakCelsius where no step ran or the cell is not solid.
    const std::vector<float>& peakCelsius() const { return peakValues; }
    // Median luminance (CIE Y) of the texels the solid cells sample; cookedLab's referenceY.
    float referenceLuminance() const { return referenceY; }

private:
    struct GpuImage{
        vk::raii::Image image = nullptr;
        vk::raii::DeviceMemory memory = nullptr;
        vk::raii::ImageView view = nullptr;
    };

    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                                                                     vk::MemoryPropertyFlags properties) const;
    GpuImage createImage(vk::ImageType type, vk::Format format, vk::Extent3D extent, vk::ImageUsageFlags usage,
                         vk::ImageViewType viewType) const;
    vk::raii::CommandBuffer beginCommands() const;
    void submit(vk::raii::CommandBuffer& commandBuffer) const;
    void dispatch(vk::raii::CommandBuffer& commandBuffer, vk::Pipeline pipeline, vk::PipelineLayout layout,
                  vk::DescriptorSet set) const;

    const vk::raii::Device& device;
    const vk::raii::PhysicalDevice& physicalDevice;
    const vk::raii::Queue& queue;

    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::DescriptorSetLayout initLayout = nullptr;
    vk::raii::DescriptorSetLayout kineticsLayout = nullptr;
    vk::raii::DescriptorSetLayout exportLayout = nullptr;
    vk::raii::PipelineLayout initPipelineLayout = nullptr;
    vk::raii::PipelineLayout kineticsPipelineLayout = nullptr;
    vk::raii::PipelineLayout exportPipelineLayout = nullptr;
    vk::raii::Pipeline initPipeline = nullptr;
    vk::raii::Pipeline kineticsPipeline = nullptr;
    vk::raii::Pipeline exportPipeline = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSet initSet = nullptr;
    vk::raii::DescriptorSet kineticsSet[2] = {nullptr, nullptr};
    vk::raii::DescriptorSet exportSet[2] = {nullptr, nullptr};
    vk::raii::DescriptorSet exportInitialSet = nullptr;
    vk::raii::Sampler albedoSampler = nullptr;

    GpuImage albedo;
    GpuImage lab0;
    GpuImage labImage[2];
    GpuImage peakImage;
    vk::raii::Buffer cellBuffer = nullptr;
    vk::raii::DeviceMemory cellMemory = nullptr;
    vk::raii::Buffer uvBuffer = nullptr;
    vk::raii::DeviceMemory uvMemory = nullptr;
    vk::raii::Buffer exportBuffer = nullptr;
    vk::raii::DeviceMemory exportMemory = nullptr;
    vk::raii::Buffer initialExportBuffer = nullptr;
    vk::raii::DeviceMemory initialExportMemory = nullptr;
    vk::raii::Buffer peakExportBuffer = nullptr;
    vk::raii::DeviceMemory peakExportMemory = nullptr;

    glm::uvec3 dims{};
    float dt = 1.0f;
    float referenceY = 0.0f;
    uint32_t completedSteps = 0;
    std::vector<glm::vec4> labValues;
    std::vector<glm::vec3> rgbValues;
    std::vector<glm::vec3> initialRgbValues;
    std::vector<float> peakValues;
};
