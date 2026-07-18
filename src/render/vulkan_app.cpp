#include "render/vulkan_app.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

struct QueueFamilies {
  std::optional<uint32_t> graphics;
  std::optional<uint32_t> present;

  bool complete() const { return graphics.has_value() && present.has_value(); }
};

struct SwapchainSupport {
  VkSurfaceCapabilitiesKHR capabilities{};
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

std::vector<char> read_file(const char* path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file) {
    throw std::runtime_error(std::string("failed to open ") + path);
  }
  const std::size_t size = file.tellg();
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(size));
  return buffer;
}

QueueFamilies find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface) {
  QueueFamilies families;
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());

  for (uint32_t i = 0; i < count; ++i) {
    if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      families.graphics = i;
    }
    VkBool32 present_support = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
    if (present_support) {
      families.present = i;
    }
  }
  return families;
}

SwapchainSupport query_swapchain_support(VkPhysicalDevice device, VkSurfaceKHR surface) {
  SwapchainSupport support;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &support.capabilities);

  uint32_t count = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
  support.formats.resize(count);
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, support.formats.data());

  count = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
  support.present_modes.resize(count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count,
                                            support.present_modes.data());
  return support;
}

VkShaderModule create_shader_module(VkDevice device, const std::vector<char>& code) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
  VkShaderModule module = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module");
  }
  return module;
}

}  // namespace

struct VulkanSliceViewer::Impl {
  int width = 0;
  int height = 0;
  GLFWwindow* window = nullptr;

  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphics_queue = VK_NULL_HANDLE;
  VkQueue present_queue = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  std::vector<VkImage> swapchain_images;
  std::vector<VkImageView> swapchain_image_views;
  VkFormat swapchain_format = VK_FORMAT_UNDEFINED;
  VkExtent2D swapchain_extent{};
  VkRenderPass render_pass = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  std::vector<VkFramebuffer> framebuffers;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  VkSemaphore image_available = VK_NULL_HANDLE;
  VkSemaphore render_finished = VK_NULL_HANDLE;
  VkFence in_flight = VK_NULL_HANDLE;

  VkSampler sampler = VK_NULL_HANDLE;
  VkImage texture_image = VK_NULL_HANDLE;
  VkDeviceMemory texture_memory = VK_NULL_HANDLE;
  VkImageView texture_view = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

  QueueFamilies queue_families{};

  ~Impl() { cleanup(); }

  void cleanup() {
    if (device) {
      vkDeviceWaitIdle(device);
      if (texture_view) vkDestroyImageView(device, texture_view, nullptr);
      if (texture_image) vkDestroyImage(device, texture_image, nullptr);
      if (texture_memory) vkFreeMemory(device, texture_memory, nullptr);
      if (sampler) vkDestroySampler(device, sampler, nullptr);
      if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
      if (descriptor_layout) vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
      if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
      if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
      if (render_pass) vkDestroyRenderPass(device, render_pass, nullptr);
      for (auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
      for (auto view : swapchain_image_views) vkDestroyImageView(device, view, nullptr);
      if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
      if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
      if (in_flight) vkDestroyFence(device, in_flight, nullptr);
      if (image_available) vkDestroySemaphore(device, image_available, nullptr);
      if (render_finished) vkDestroySemaphore(device, render_finished, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    if (window) {
      glfwDestroyWindow(window);
      glfwTerminate();
    }
  }

  void init(int w, int h, const char* title) {
    width = w;
    height = h;
    if (!glfwInit()) {
      throw std::runtime_error("glfwInit failed");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
      throw std::runtime_error("glfwCreateWindow failed");
    }

    uint32_t ext_count = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&ext_count);

    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.pApplicationName = "steak-sim";
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app_info;
    instance_info.enabledExtensionCount = ext_count;
    instance_info.ppEnabledExtensionNames = extensions;

    if (vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateInstance failed");
    }

    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
      throw std::runtime_error("glfwCreateWindowSurface failed");
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    if (device_count == 0) {
      throw std::runtime_error("no Vulkan physical devices");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
    physical_device = devices[0];

    queue_families = find_queue_families(physical_device, surface);
    if (!queue_families.complete()) {
      throw std::runtime_error("incomplete queue families");
    }

    std::set<uint32_t> unique_queues = {queue_families.graphics.value(),
                                        queue_families.present.value()};
    float priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    for (uint32_t family : unique_queues) {
      VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
      qi.queueFamilyIndex = family;
      qi.queueCount = 1;
      qi.pQueuePriorities = &priority;
      queue_infos.push_back(qi);
    }

    const std::array<const char*, 1> device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
    device_info.pQueueCreateInfos = queue_infos.data();
    device_info.pEnabledFeatures = &features;
    device_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    device_info.ppEnabledExtensionNames = device_extensions.data();

    if (vkCreateDevice(physical_device, &device_info, nullptr, &device) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateDevice failed");
    }

    vkGetDeviceQueue(device, queue_families.graphics.value(), 0, &graphics_queue);
    vkGetDeviceQueue(device, queue_families.present.value(), 0, &present_queue);

    create_swapchain();
    create_image_views();
    create_render_pass();
    create_pipeline();
    create_framebuffers();
    create_command_pool();
    create_sync_objects();
    create_texture_resources();
    create_command_buffer();
  }

  void create_swapchain() {
    const SwapchainSupport support = query_swapchain_support(physical_device, surface);
    VkSurfaceFormatKHR surface_format = support.formats[0];
    for (const auto& format : support.formats) {
      if (format.format == VK_FORMAT_R8G8B8A8_UNORM) {
        surface_format = format;
        break;
      }
    }

    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D extent = support.capabilities.currentExtent;
    if (extent.width == UINT32_MAX) {
      extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }

    uint32_t image_count = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 &&
        image_count > support.capabilities.maxImageCount) {
      image_count = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = image_count;
    ci.imageFormat = surface_format.format;
    ci.imageColorSpace = surface_format.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    const uint32_t queue_indices[] = {queue_families.graphics.value(),
                                      queue_families.present.value()};
    if (queue_families.graphics != queue_families.present) {
      ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      ci.queueFamilyIndexCount = 2;
      ci.pQueueFamilyIndices = queue_indices;
    } else {
      ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform = support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = present_mode;
    ci.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateSwapchainKHR failed");
    }

    vkGetSwapchainImagesKHR(device, swapchain, &image_count, nullptr);
    swapchain_images.resize(image_count);
    vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.data());
    swapchain_format = surface_format.format;
    swapchain_extent = extent;
  }

  void create_image_views() {
    swapchain_image_views.resize(swapchain_images.size());
    for (std::size_t i = 0; i < swapchain_images.size(); ++i) {
      VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      ci.image = swapchain_images[i];
      ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      ci.format = swapchain_format;
      ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      ci.subresourceRange.levelCount = 1;
      ci.subresourceRange.layerCount = 1;
      if (vkCreateImageView(device, &ci, nullptr, &swapchain_image_views[i]) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateImageView failed");
      }
    }
  }

  void create_render_pass() {
    VkAttachmentDescription color{};
    color.format = swapchain_format;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 1;
    ci.pAttachments = &color;
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;

    if (vkCreateRenderPass(device, &ci, nullptr, &render_pass) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateRenderPass failed");
    }
  }

  void create_pipeline() {
    const auto vert_code = read_file(STEAK_SHADER_DIR "/slice.vert.spv");
    const auto frag_code = read_file(STEAK_SHADER_DIR "/slice.frag.spv");
    VkShaderModule vert = create_shader_module(device, vert_code);
    VkShaderModule frag = create_shader_module(device, frag_code);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo input_assembly{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.width = static_cast<float>(swapchain_extent.width);
    viewport.height = static_cast<float>(swapchain_extent.height);
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, swapchain_extent};
    VkPipelineViewportStateCreateInfo viewport_state{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout_ci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_ci.bindingCount = 1;
    layout_ci.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(device, &layout_ci, nullptr, &descriptor_layout) !=
        VK_SUCCESS) {
      throw std::runtime_error("vkCreateDescriptorSetLayout failed");
    }

    VkPipelineLayoutCreateInfo pipeline_layout_ci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_ci.setLayoutCount = 1;
    pipeline_layout_ci.pSetLayouts = &descriptor_layout;
    if (vkCreatePipelineLayout(device, &pipeline_layout_ci, nullptr, &pipeline_layout) !=
        VK_SUCCESS) {
      throw std::runtime_error("vkCreatePipelineLayout failed");
    }

    VkGraphicsPipelineCreateInfo pipeline_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline_ci.stageCount = 2;
    pipeline_ci.pStages = stages;
    pipeline_ci.pVertexInputState = &vertex_input;
    pipeline_ci.pInputAssemblyState = &input_assembly;
    pipeline_ci.pViewportState = &viewport_state;
    pipeline_ci.pRasterizationState = &raster;
    pipeline_ci.pMultisampleState = &multisample;
    pipeline_ci.pColorBlendState = &blend;
    pipeline_ci.layout = pipeline_layout;
    pipeline_ci.renderPass = render_pass;
    pipeline_ci.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr,
                                  &pipeline) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateGraphicsPipelines failed");
    }

    vkDestroyShaderModule(device, frag, nullptr);
    vkDestroyShaderModule(device, vert, nullptr);
  }

  void create_framebuffers() {
    framebuffers.resize(swapchain_image_views.size());
    for (std::size_t i = 0; i < swapchain_image_views.size(); ++i) {
      VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      ci.renderPass = render_pass;
      ci.attachmentCount = 1;
      ci.pAttachments = &swapchain_image_views[i];
      ci.width = swapchain_extent.width;
      ci.height = swapchain_extent.height;
      ci.layers = 1;
      if (vkCreateFramebuffer(device, &ci, nullptr, &framebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("vkCreateFramebuffer failed");
      }
    }
  }

  void create_command_pool() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = queue_families.graphics.value();
    if (vkCreateCommandPool(device, &ci, nullptr, &command_pool) != VK_SUCCESS) {
      throw std::runtime_error("vkCreateCommandPool failed");
    }
  }

  void create_sync_objects() {
    VkSemaphoreCreateInfo sem_ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fence_ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateSemaphore(device, &sem_ci, nullptr, &image_available);
    vkCreateSemaphore(device, &sem_ci, nullptr, &render_finished);
    vkCreateFence(device, &fence_ci, nullptr, &in_flight);
  }

  void create_texture_resources() {
    VkSamplerCreateInfo sampler_ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_ci.magFilter = VK_FILTER_LINEAR;
    sampler_ci.minFilter = VK_FILTER_LINEAR;
    sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device, &sampler_ci, nullptr, &sampler);

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = 1;
    VkDescriptorPoolCreateInfo pool_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &pool_size;
    vkCreateDescriptorPool(device, &pool_ci, nullptr, &descriptor_pool);

    VkDescriptorSetAllocateInfo alloc_ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc_ci.descriptorPool = descriptor_pool;
    alloc_ci.descriptorSetCount = 1;
    alloc_ci.pSetLayouts = &descriptor_layout;
    vkAllocateDescriptorSets(device, &alloc_ci, &descriptor_set);
  }

  void create_command_buffer() {
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = command_pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &alloc, &command_buffer);
  }

  void upload_slice(const SliceImage& slice) {
    if (texture_image) {
      vkDestroyImageView(device, texture_view, nullptr);
      vkDestroyImage(device, texture_image, nullptr);
      vkFreeMemory(device, texture_memory, nullptr);
      texture_view = VK_NULL_HANDLE;
      texture_image = VK_NULL_HANDLE;
      texture_memory = VK_NULL_HANDLE;
    }

    VkImageCreateInfo image_ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_ci.imageType = VK_IMAGE_TYPE_2D;
    image_ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_ci.extent = {static_cast<uint32_t>(slice.width),
                       static_cast<uint32_t>(slice.height), 1};
    image_ci.mipLevels = 1;
    image_ci.arrayLayers = 1;
    image_ci.samples = VK_SAMPLE_COUNT_1_BIT;
    image_ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(device, &image_ci, nullptr, &texture_image);

    VkMemoryRequirements mem_req{};
    vkGetImageMemoryRequirements(device, texture_image, &mem_req);
    VkMemoryAllocateInfo alloc_ci{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc_ci.allocationSize = mem_req.size;
    alloc_ci.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &alloc_ci, nullptr, &texture_memory);
    vkBindImageMemory(device, texture_image, texture_memory, 0);

    const VkDeviceSize image_size = slice.rgba.size();
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;

    VkBufferCreateInfo buffer_ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_ci.size = image_size;
    buffer_ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkCreateBuffer(device, &buffer_ci, nullptr, &staging_buffer);
    vkGetBufferMemoryRequirements(device, staging_buffer, &mem_req);
    alloc_ci.allocationSize = mem_req.size;
    alloc_ci.memoryTypeIndex =
        find_memory_type(mem_req.memoryTypeBits,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &alloc_ci, nullptr, &staging_memory);
    vkBindBufferMemory(device, staging_buffer, staging_memory, 0);

    void* mapped = nullptr;
    vkMapMemory(device, staging_memory, 0, image_size, 0, &mapped);
    std::memcpy(mapped, slice.rgba.data(), image_size);
    vkUnmapMemory(device, staging_memory);

    transition_image(texture_image, VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copy_buffer_to_image(staging_buffer, texture_image, slice.width, slice.height);
    transition_image(texture_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, staging_buffer, nullptr);
    vkFreeMemory(device, staging_memory, nullptr);

    VkImageViewCreateInfo view_ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_ci.image = texture_image;
    view_ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_ci.subresourceRange.levelCount = 1;
    view_ci.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &view_ci, nullptr, &texture_view);

    VkDescriptorImageInfo image_info{};
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView = texture_view;
    image_info.sampler = sampler;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptor_set;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &image_info;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
  }

  uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
      if ((type_bits & (1u << i)) &&
          (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
        return i;
      }
    }
    throw std::runtime_error("failed to find memory type");
  }

  void transition_image(VkImage image, VkImageLayout old_layout, VkImageLayout new_layout) {
    VkCommandBuffer cmd = begin_single_time_commands();
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
        new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    end_single_time_commands(cmd);
  }

  void copy_buffer_to_image(VkBuffer buffer, VkImage image, int width, int height) {
    VkCommandBuffer cmd = begin_single_time_commands();
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);
    end_single_time_commands(cmd);
  }

  VkCommandBuffer begin_single_time_commands() {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = command_pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &alloc, &cmd);

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
  }

  void end_single_time_commands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    vkQueueSubmit(graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue);
    vkFreeCommandBuffers(device, command_pool, 1, &cmd);
  }

  bool draw_frame() {
    vkWaitForFences(device, 1, &in_flight, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &in_flight);

    uint32_t image_index = 0;
    VkResult acquire = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_available,
                                             VK_NULL_HANDLE, &image_index);
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
      return false;
    }

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(command_buffer, &begin);

    VkClearValue clear_color{{{0.02f, 0.02f, 0.05f, 1.0f}}};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass = render_pass;
    rp.framebuffer = framebuffers[image_index];
    rp.renderArea.extent = swapchain_extent;
    rp.clearValueCount = 1;
    rp.pClearValues = &clear_color;

    vkCmdBeginRenderPass(command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                            0, 1, &descriptor_set, 0, nullptr);
    vkCmdDraw(command_buffer, 3, 1, 0, 0);
    vkCmdEndRenderPass(command_buffer);
    vkEndCommandBuffer(command_buffer);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &image_available;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_buffer;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &render_finished;
    vkQueueSubmit(graphics_queue, 1, &submit, in_flight);

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &render_finished;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &image_index;
    vkQueuePresentKHR(present_queue, &present);
    return true;
  }
};

VulkanSliceViewer::VulkanSliceViewer(int width, int height, const char* title)
    : impl_(new Impl) {
  try {
    impl_->init(width, height, title);
    ok_ = true;
  } catch (const std::exception& ex) {
    error_ = ex.what();
  }
}

VulkanSliceViewer::~VulkanSliceViewer() { delete impl_; }

bool VulkanSliceViewer::pump_once(const SliceImage& slice) {
  if (!ok_ || glfwWindowShouldClose(impl_->window)) {
    return false;
  }
  impl_->upload_slice(slice);
  impl_->draw_frame();
  glfwPollEvents();
  return true;
}

bool verify_vulkan_instance() {
  if (!glfwInit()) {
    return false;
  }
  uint32_t ext_count = 0;
  const char** extensions = glfwGetRequiredInstanceExtensions(&ext_count);
  VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app_info.pApplicationName = "steak-sim";
  app_info.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &app_info;
  ci.enabledExtensionCount = ext_count;
  ci.ppEnabledExtensionNames = extensions;
  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) {
    glfwTerminate();
    return false;
  }
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  vkDestroyInstance(instance, nullptr);
  glfwTerminate();
  return device_count > 0;
}

SliceImage make_temperature_slice(const std::vector<float>& grid, int nx, int ny, int nz,
                                  int z_layer, float min_temp, float max_temp) {
  SliceImage image;
  image.width = nx;
  image.height = ny;
  image.rgba.resize(static_cast<std::size_t>(nx * ny * 4));
  const float denom = std::max(max_temp - min_temp, 1e-3f);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int idx = x + nx * (y + ny * z_layer);
      const float norm = (grid[idx] - min_temp) / denom;
      const std::size_t px = static_cast<std::size_t>((y * nx + x) * 4);
      image.rgba[px + 0] = static_cast<uint8_t>(std::clamp(norm, 0.0f, 1.0f) * 255.0f);
      image.rgba[px + 1] = 0;
      image.rgba[px + 2] = 0;
      image.rgba[px + 3] = 255;
    }
  }
  return image;
}
