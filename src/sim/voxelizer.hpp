#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

constexpr uint32_t VOXEL_INSIDE = 1u;   // voxel center is inside the closed mesh
constexpr uint32_t VOXEL_SURFACE = 2u;  // voxel box touches at least one triangle

struct VoxelGrid{
    glm::vec3 origin{};
    float voxelSize = 0.0f;
    glm::uvec3 dims{};
    std::vector<uint32_t> cells;  // VOXEL_* flags; index = x + nx * (y + ny * z)

    size_t cellCount() const { return size_t(dims.x) * dims.y * dims.z; }
    uint32_t at(uint32_t x, uint32_t y, uint32_t z) const { return cells[x + dims.x * (y + dims.y * z)]; }
};

// Grid snapped to multiples of voxelSize around [low, high], with one empty voxel of
// padding per side. Leaves cells empty.
VoxelGrid makeVoxelGridLayout(glm::vec3 low, glm::vec3 high, float voxelSize);

// Runs shaders/compute/voxelize.slang. The device, physical device and queue must outlive it.
class Voxelizer{
public:
    Voxelizer(const vk::raii::Device& device, const vk::raii::PhysicalDevice& physicalDevice,
              const vk::raii::Queue& queue, uint32_t queueFamilyIndex, const std::string& shaderPath);

    // Triangles already on the GPU; each vertex's position is its first three floats.
    // Both buffers need eStorageBuffer usage.
    VoxelGrid voxelize(vk::Buffer vertexBuffer, uint32_t vertexStrideFloats, vk::Buffer indexBuffer,
                       uint32_t triangleCount, glm::vec3 low, glm::vec3 high, float voxelSize);

    // Uploads the mesh to the GPU, then voxelizes it.
    VoxelGrid voxelize(const std::vector<glm::vec3>& positions, const std::vector<uint32_t>& indices,
                       float voxelSize);

private:
    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createHostBuffer(vk::DeviceSize size,
                                                                       vk::BufferUsageFlags usage) const;

    const vk::raii::Device& device;
    const vk::raii::PhysicalDevice& physicalDevice;
    const vk::raii::Queue& queue;
    vk::raii::CommandPool commandPool = nullptr;
    vk::raii::DescriptorSetLayout setLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    vk::raii::DescriptorSet descriptorSet = nullptr;
};
