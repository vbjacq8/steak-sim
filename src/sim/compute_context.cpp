#include "compute_context.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

ComputeContext::ComputeContext(const char* appName){
    std::vector<const char*> instanceExtensions;
    vk::InstanceCreateFlags instanceFlags{};
    for (const auto& extension : context.enumerateInstanceExtensionProperties()){
        // MoltenVK is only enumerated when portability enumeration is requested.
        if (std::strcmp(extension.extensionName, vk::KHRPortabilityEnumerationExtensionName) == 0){
            instanceExtensions.push_back(vk::KHRPortabilityEnumerationExtensionName);
            instanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
        }
    }
    vk::ApplicationInfo appInfo{.pApplicationName = appName, .apiVersion = vk::ApiVersion13};
    instance = vk::raii::Instance(context, vk::InstanceCreateInfo{
        .flags = instanceFlags,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
        .ppEnabledExtensionNames = instanceExtensions.data()});

    for (auto& candidate : instance.enumeratePhysicalDevices()){
        if (candidate.getProperties().apiVersion < vk::ApiVersion13) continue;
        const auto families = candidate.getQueueFamilyProperties();
        for (uint32_t i = 0; i < families.size(); ++i){
            if (families[i].queueFlags & vk::QueueFlagBits::eCompute){
                physicalDevice = std::move(candidate);
                queueFamily = i;
                break;
            }
        }
        if (*physicalDevice) break;
    }
    if (!*physicalDevice) throw std::runtime_error("no Vulkan 1.3 device with a compute queue");

    std::vector<const char*> deviceExtensions;
    for (const auto& extension : physicalDevice.enumerateDeviceExtensionProperties()){
        if (std::strcmp(extension.extensionName, "VK_KHR_portability_subset") == 0){
            deviceExtensions.push_back("VK_KHR_portability_subset");
        }
    }
    const float priority = 1.0f;
    vk::DeviceQueueCreateInfo queueInfo{.queueFamilyIndex = queueFamily, .queueCount = 1, .pQueuePriorities = &priority};
    vk::PhysicalDeviceVulkan13Features features13{.synchronization2 = true};
    device = vk::raii::Device(physicalDevice, vk::DeviceCreateInfo{
        .pNext = &features13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data()});
    queue = vk::raii::Queue(device, queueFamily, 0);
}
