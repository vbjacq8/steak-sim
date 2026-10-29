#pragma once

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>

// Headless Vulkan 1.3 device with one compute queue, for running compute shaders without a window.
struct ComputeContext{
    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue queue = nullptr;
    uint32_t queueFamily = 0;

    explicit ComputeContext(const char* appName);
};
