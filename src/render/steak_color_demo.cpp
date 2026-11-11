// Steak mesh viewer. Same load path as steak_sim, with the surface color cooked in the fragment
// shader. No voxel grid. Camera controls match the Open3D viewers:
//
//   Left drag          rotate
//   Right drag         pan
//   Shift+left drag    zoom
//   Scroll             zoom
//   Space              raw texture / cooked color

#include "vulkan/vulkan.hpp"
#include "vulkan/vulkan_core.h"
#include <cstdint>
#include <limits>
#include <sys/syslimits.h>
#include <sys/types.h>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include<vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include<GLFW/glfw3.h>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>

#include<algorithm>
#include<chrono>
#include<cmath>
#include<iostream>
#include<fstream>
#include<stdexcept>
#include<cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include<stb_image.h>

#include "mesh_loader.hpp"

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
constexpr uint32_t FRAMES_IN_FLIGHT = 2;
const std::string MODEL_PATH = STEAK_OBJ;
const std::string TEXTURE_PATH = STEAK_TEXTURE;

const std::vector<char const *>  validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription(){
        return {.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions(){
        return {{{.location = 0, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, pos)},
                {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)},
                {.location = 2, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, texCoord)}
            }};
    }
};

struct UniformBufferObject{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    // x = seconds the surface has been cooking, y = temperature in degrees Fahrenheit.
    // One temperature for the whole mesh: this demo does not voxelize.
    alignas(16) glm::vec4 cook;
};

// Hot enough that the cooked state is visibly different from the raw texture.
constexpr float COOK_TEMPERATURE_F = 212.0f;
// Long enough at that temperature for a* to sit on the equilibrium color.
constexpr float COOK_SECONDS = 240.0f;

/**  previous hard-coded vertex and index arrays for vulkan tutorial texture / depth load
std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};


std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4
};

*/

class HelloTriangleApplication{
public:
    void run(){
        initWindow();
        init();
        std::cout << "Left drag: rotate\n"
                     "Right drag: pan\n"
                     "Shift+left drag or scroll: zoom\n"
                     "Space: raw texture / cooked color\n";
        mainLoop();
        cleanUp();
    }

private:
    void init(){
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResources();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void loadModel(){
        const Mesh mesh = loadMesh(MODEL_PATH);
        // OBJ stores V pointing up. The uploaded image is top row first, so flip OBJ V only.
        const bool flipV = mesh.format == MeshFormat::Obj;
        vertices.resize(mesh.positions.size());
        for (size_t i = 0; i < mesh.positions.size(); ++i){
            Vertex vertex{};
            vertex.pos = mesh.positions[i];
            vertex.color = {1.0f, 1.0f, 1.0f};
            if (!mesh.texcoords.empty()){
                vertex.texCoord = mesh.texcoords[i];
                if (flipV) vertex.texCoord.y = 1.0f - vertex.texCoord.y;
            }
            vertices[i] = vertex;
        }
        indices = mesh.indices;
    }

    void createDepthResources(){
        vk::Format depthFormat = findDepthFormat();
        std::tie(depthImage, depthImageMemory) = createImage(
            swapchainExtent.width, 
            swapchainExtent.height, 
            depthFormat,
            vk::ImageTiling::eOptimal, 
            vk::ImageUsageFlagBits::eDepthStencilAttachment, 
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);

    }

    void createTextureSampler(){
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,

            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .mipLodBias = 0.0f,
            .minLod = 0.0f,
            .maxLod = 0.0f,

            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,

            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,

            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,

            .borderColor = vk::BorderColor::eIntOpaqueBlack,

            .unnormalizedCoordinates = vk::False
        };

        textureSampler = vk::raii::Sampler(device, samplerInfo);

    }


    void createTextureImage(){
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;
        //pointer that is returned is the first element of the array of pixel values, 4 bytes per pixel
        if (!pixels){
            throw std::runtime_error("failed to grab texture");
        }

        auto [stagingBuffer, stagingBufferMemory] = createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);

        void* data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, imageSize);
        stagingBufferMemory.unmapMemory();

        stbi_image_free(pixels);

        std::tie(textureImage, textureImageMemory) = createImage(
            texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal, 
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, 
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
        transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(commandBuffer, stagingBuffer, textureImage, static_cast<int>(texWidth), static_cast<int>(texHeight));
        transitionImageLayout(commandBuffer, textureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        endSingleTimeCommands(std::move(commandBuffer));

    }

    std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(
        uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, 
        vk::MemoryPropertyFlags properties)
    {
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        vk::raii::Image image = vk::raii::Image(device, imageInfo);
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
        };
        vk::raii::DeviceMemory imageMemory = vk::raii::DeviceMemory(device, allocInfo);
        image.bindMemory(imageMemory, 0);

        return {std::move(image), std::move(imageMemory)};
        
    }

    vk::raii::CommandBuffer beginSingleTimeCommands(){
        vk::CommandBufferAllocateInfo bufferAllocInfo{
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary, 
            .commandBufferCount = 1,
        };
        vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, bufferAllocInfo).front());
        vk::CommandBufferBeginInfo commandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        commandBuffer.begin(commandBufferBeginInfo);
        return std::move(commandBuffer);
    }

    void endSingleTimeCommands(vk::raii::CommandBuffer &&commandBuffer){
        commandBuffer.end();
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer
        };
        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    void createDescriptorSets(){
        std::vector<vk::DescriptorSetLayout> layouts(FRAMES_IN_FLIGHT, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data()
        };

        descriptorSets = device.allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < FRAMES_IN_FLIGHT; ++i){
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject)
            };

            vk::DescriptorImageInfo imageInfo{
                .sampler = textureSampler,
                .imageView = textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
            };

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{{
                {
                .dstSet = descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo
                },
                {.dstSet = descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo,
                }
            }};

            device.updateDescriptorSets(descriptorWrites, {});



        }
    }

    void createDescriptorPool(){
        std::array<vk::DescriptorPoolSize, 2> poolSize{{
            {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = FRAMES_IN_FLIGHT},
            {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = FRAMES_IN_FLIGHT}
        }};
        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = FRAMES_IN_FLIGHT, 
            .poolSizeCount = static_cast<uint32_t>(poolSize.size()), 
            .pPoolSizes = poolSize.data()
        };

        descriptorPool = vk::raii::DescriptorPool(device, poolInfo);


    }

    void createDescriptorSetLayout(){
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
            {.binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex
            },

            {.binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment
            }
        }};

        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings = bindings.data()
        };

        descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    }

    // Camera frame matches Open3D: columns are right, up, and back (+Z is backward).
    void lookAtPose(glm::vec3 eyeIn, glm::vec3 targetIn, glm::vec3 upHint){
        eye = eyeIn;
        target = targetIn;
        const glm::vec3 forward = glm::normalize(target - eye);
        glm::vec3 right = glm::cross(forward, upHint);
        if (glm::dot(right, right) < 1e-8f) right = glm::cross(forward, glm::vec3(0.0f, 0.0f, 1.0f));
        right = glm::normalize(right);
        const glm::vec3 up = glm::normalize(glm::cross(right, forward));
        camRot = glm::mat3(right, up, -forward);
    }

    glm::mat4 viewMatrix() const{
        const glm::vec3 forward = -camRot[2];
        return glm::lookAt(eye, eye + forward, camRot[1]);
    }

    float viewHeight() const{
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        return static_cast<float>(std::max(height, 1));
    }

    void beginDrag(double x, double y){
        downX = x;
        downY = y;
        eyeDown = eye;
        targetDown = target;
        rotDown = camRot;
    }

    // Open3D MatrixInteractorLogic::Rotate: trackball about the center of rotation.
    void applyRotate(float dx, float dy){
        dy = -dy;
        const glm::vec3 moved(dx, dy, 0.0f);
        const float movedLen = glm::length(moved);
        if (movedLen < 1e-4f){
            eye = eyeDown;
            camRot = rotDown;
            target = targetDown;
            return;
        }
        glm::vec3 axis = glm::normalize(glm::vec3(-dy, dx, 0.0f));
        const float theta = 3.14159265f * movedLen / viewHeight();
        axis = rotDown * axis;
        const glm::mat3 spin = glm::mat3(glm::rotate(glm::mat4(1.0f), -theta, axis));
        camRot = spin * rotDown;

        glm::vec3 toCenter = targetDown - eyeDown;
        float dist = glm::length(toCenter);
        const glm::vec3 forward = rotDown * glm::vec3(0.0f, 0.0f, -1.0f);
        if (glm::dot(toCenter, forward) < 0.0f) dist = -dist;
        eye = targetDown + camRot * glm::vec3(0.0f, 0.0f, dist);
        target = targetDown;
    }

    // Open3D RotationInteractorLogic::Pan. dx/dy are pixels from the drag start.
    void applyPan(float dx, float dy){
        const glm::vec3 forward = rotDown * glm::vec3(0.0f, 0.0f, -1.0f);
        float dist = glm::dot(forward, targetDown - eyeDown);
        dist = std::max(0.1f, dist);
        const float unitsPerPixel = 2.0f * std::tan(glm::radians(22.5f)) * (0.1f + dist) / viewHeight();
        const glm::vec3 worldMove = rotDown * glm::vec3(-dx * unitsPerPixel, dy * unitsPerPixel, 0.0f);
        eye = eyeDown + worldMove;
        target = targetDown + worldMove;
        camRot = rotDown;
    }

    // Open3D MatrixInteractorLogic::Dolly for a vertical mouse drag.
    void applyDolly(float dy){
        const float length = std::max(0.5f, glm::length(targetDown - eyeDown));
        const float zDist = dy * 0.0025f * length;
        eye = eyeDown + rotDown * glm::vec3(0.0f, 0.0f, -zDist);
        target = targetDown;
        camRot = rotDown;
    }

    void applyRoll(float dy){
        const float radians = 4.0f * 3.14159265f * dy / viewHeight();
        camRot = rotDown * glm::mat3(glm::rotate(glm::mat4(1.0f), radians, glm::vec3(0.0f, 0.0f, 1.0f)));
        eye = eyeDown;
        target = targetDown;
    }

    void applyScroll(float dy){
        const float length = std::max(0.5f, glm::length(target - eye));
        const float zDist = -dy * 0.05f * length;
        eye += camRot * glm::vec3(0.0f, 0.0f, -zDist);
    }

    void handleInput(){
        double cursorX = 0.0, cursorY = 0.0;
        glfwGetCursorPos(window, &cursorX, &cursorY);
        const bool left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        const bool right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        const bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                           glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        const bool alt = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                         glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
        const bool ctrl = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                          glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
        const bool meta = glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
                          glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS;

        if (drag == Drag::None && (left || right) && !(leftWasDown || rightWasDown)){
            if (right || (left && ctrl)) drag = Drag::Pan;
            else if (left && meta) drag = Drag::Roll;
#ifdef __APPLE__
            else if (left && shift && alt) drag = Drag::Roll;
#else
            else if (left && shift && ctrl) drag = Drag::Roll;
#endif
            else if (left && shift) drag = Drag::Dolly;
            else drag = Drag::Rotate;
            beginDrag(cursorX, cursorY);
        }
        if (drag != Drag::None && !left && !right) drag = Drag::None;

        if (drag != Drag::None){
            const float dx = static_cast<float>(cursorX - downX);
            const float dy = static_cast<float>(cursorY - downY);
            if (drag == Drag::Rotate) applyRotate(dx, dy);
            else if (drag == Drag::Pan) applyPan(dx, dy);
            else if (drag == Drag::Dolly) applyDolly(dy);
            else if (drag == Drag::Roll) applyRoll(dy);
        }
        leftWasDown = left;
        rightWasDown = right;

        const bool space = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        if (space && !spaceWasDown){
            showCooked = !showCooked;
            glfwSetWindowTitle(window, showCooked ? "steak color — after" : "steak color — before");
        }
        spaceWasDown = space;
    }

    void updateUniformBuffer(uint32_t currentImage){
        handleInput();
        const float aspect = static_cast<float>(swapchainExtent.width) / static_cast<float>(std::max(swapchainExtent.height, 1u));
        UniformBufferObject ubo{
            .model = glm::mat4(1.0f),
            .view = viewMatrix(),
            .proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f),
            .cook = glm::vec4(showCooked ? COOK_SECONDS : 0.0f, COOK_TEMPERATURE_F, 0.0f, 0.0f)
        };

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void createUniformBuffers(){
        for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++){
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            auto [buffer, bufferMem] = createBuffer(
                bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible
            );
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, bufferSize));
        }
    }

    void createInstance(){

        std::vector<char const*> requiredLayers;
        if (enableValidationLayers) requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        
        auto layerProperties = context.enumerateInstanceLayerProperties();
        auto unsupportedLayerIt = std::ranges::find_if(
            //iterate through requiredlayers
            requiredLayers, [&layerProperties](auto const &requiredLayer){
                //lambda returns true if atleast one requiredLayer is not matched, returns the unsupportedLayer
                return std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty)
                    {
                    return strcmp(layerProperty.layerName, requiredLayer) == 0;
                    }
                );
            }
        );

        if (unsupportedLayerIt != requiredLayers.end()){
            throw std::runtime_error("Required Layer not supported: " + std::string(*unsupportedLayerIt));
        }


        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName = "Hello Triangle",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1,0,0),
            .apiVersion = vk::ApiVersion14
        };

        auto glfwExtensions = getRequiredInstanceExtensions();

        std::vector<vk::ExtensionProperties> extensionProperties = context.enumerateInstanceExtensionProperties();
        auto unsupportedPropertyIt = std::ranges::find_if(glfwExtensions,
            [&extensionProperties](auto const &glfwExtension){
                return std::ranges::none_of(extensionProperties, 
                    [glfwExtension](auto const &extensionProperty){
                        return strcmp(glfwExtension, extensionProperty.extensionName) == 0;
                    }
                );
            }
        );

        if (unsupportedPropertyIt != glfwExtensions.end()){
            throw std::runtime_error("Required GLFW extension not supported: " + std::string(*unsupportedPropertyIt));
        }
            
    

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames = requiredLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(glfwExtensions.size()),
            .ppEnabledExtensionNames = glfwExtensions.data(),
        };

        instance = vk::raii::Instance(context, createInfo);


    }

    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties){
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        vk::raii::Buffer vertexBuffer = vk::raii::Buffer(device, bufferInfo);
        vk::MemoryRequirements memRequirements = vertexBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo memoryAllocateInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,  properties)
        };

        vk::raii::DeviceMemory vertexBufferMemory = vk::raii::DeviceMemory(device, memoryAllocateInfo);
        vertexBuffer.bindMemory(*vertexBufferMemory, 0);
        return {std::move(vertexBuffer), std::move(vertexBufferMemory)};

    }

    void createVertexBuffer(){
       
        //buffer size in bytes
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, 
            vk::BufferUsageFlagBits::eTransferSrc, 
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

        

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        std::tie(vertexBuffer, vertexBufferMemory) = 
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);


        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);


    }

    void createIndexBuffer(){
       
        //buffer size in bytes
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        auto [stagingBuffer, stagingBufferMemory] = createBuffer(bufferSize, 
            vk::BufferUsageFlagBits::eTransferSrc, 
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            );

        

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        std::tie(indexBuffer, indexBufferMemory) = 
            createBuffer(bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);


        copyBuffer(stagingBuffer, indexBuffer, bufferSize);


    }

    void copyBufferToImage(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Buffer &buffer, vk::raii::Image& image, uint32_t width, uint32_t height){
        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1
            },
            .imageOffset = {0,0,0},
            .imageExtent = {width, height, 1}
        };
        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    }

    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size){
        vk::raii::CommandBuffer copyCommandBuffer = beginSingleTimeCommands();
        copyCommandBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0,0,size));
        endSingleTimeCommands(std::move(copyCommandBuffer));
    }

    void mainLoop(){
        while (!glfwWindowShouldClose(window)){
            glfwPollEvents();
            drawFrame();
        }

        device.waitIdle();

    }

    void drawFrame(){
        //debug
        // std::cerr << "In drawFrame" << std::endl;


        auto fenceResult = device.waitForFences(*inFlightFences[frameIdx], vk::True, UINT64_MAX);
            //std::cerr << "waited for fences" << std::endl;
            if (fenceResult != vk::Result::eSuccess){
                throw std::runtime_error("failed to wait for fence");
            }
            

        auto [result, imageIndex] = swapchain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIdx], nullptr);
        if (result == vk::Result::eErrorOutOfDateKHR){
            recreateSwapchain();
            return;
        }
        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR){
            assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
            throw std::runtime_error("Failed to acquire swapchain image");
        }

        //we only reset the fences if we know we are submitting work (i.e, we pass the image acquisition)
        device.resetFences(*inFlightFences[frameIdx]);

        commandBuffers[frameIdx].reset();
        recordCommandBuffer(imageIndex);

        graphicsQueue.waitIdle();

        updateUniformBuffer(frameIdx);

        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*presentCompleteSemaphores[frameIdx],
            .pWaitDstStageMask = &waitDestinationStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffers[frameIdx],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &*renderFinishedSemaphores[frameIdx]
        };

        graphicsQueue.submit(submitInfo, *inFlightFences[frameIdx]);


        const vk::PresentInfoKHR presentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &*renderFinishedSemaphores[frameIdx],
            .swapchainCount = 1,
            .pSwapchains = &*swapchain,
            .pImageIndices = &imageIndex

        };

        result = graphicsQueue.presentKHR(presentInfoKHR);
        if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR || framebufferResized){
            recreateSwapchain();
            framebufferResized = false;
        }
        else assert(result == vk::Result::eSuccess);

        frameIdx = (frameIdx + 1) % FRAMES_IN_FLIGHT;

    }


    void cleanUp(){
        cleanUpSwapchain();
        glfwDestroyWindow(window);
        glfwTerminate();

    }

    void initWindow(){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "steak color — before", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetScrollCallback(window, scrollCallback);
        lookAtPose(glm::vec3(50.0f, 2.0f, 2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    }

    //wrap the required instance extensions (originally char**) in std::vector
    std::vector<const char*> getRequiredInstanceExtensions(){
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) extensions.push_back(vk::EXTDebugUtilsExtensionName);

        return extensions;

    }

    void pickPhysicalDevice(){
        std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
        if (physicalDevices.empty()) throw std::runtime_error("No valid GPUs with Vulkan support");

        auto devIter = std::ranges::find_if(physicalDevices, [&](auto const& thisPhysicalDevice){
            return isDeviceSuitable(thisPhysicalDevice);
        });
        if (devIter == physicalDevices.end()) throw std::runtime_error("No valid GPUs with Vulkan Support");
        physicalDevice = *devIter;

    }

    bool isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice){
        /* example where we want a discrete GPU that supports geometric shading
        vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
        vk::PhysicalDeviceFeatures deviceFeatures = physicalDevice.getFeatures();

        if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && deviceFeatures.geometryShader){
            return true;
        }

        return false;
        */

        bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;
        std::vector<vk::QueueFamilyProperties> qFamilies = physicalDevice.getQueueFamilyProperties();
        bool supportsGraphics = std::ranges::any_of(qFamilies, 
            [](const auto &qFamilyProperties){
                return !!(qFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics);
            }
        );

        std::vector<const char*> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName};
        std::vector<vk::ExtensionProperties> availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
        bool supportsAllRequiredExtensions = std::ranges::all_of(
            requiredDeviceExtensions, [&availableDeviceExtensions](auto const &requiredDeviceExtension){
                return std::ranges::any_of(
                    availableDeviceExtensions, [requiredDeviceExtension](auto const &availableDeviceExtension){
                        return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
                    }
                );
            }
        );

        auto features = physicalDevice.template getFeatures2<
            vk::PhysicalDeviceFeatures2,
            //vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

        bool supportsFeatures = 
            features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
            //features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
            features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
            features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
            features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

        
        return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsFeatures;

    }

    void createLogicalDevice(){
    
        std::vector<vk::QueueFamilyProperties> qFamilies = physicalDevice.getQueueFamilyProperties();
        //unsigned qIndex = ~0;
        for (unsigned qfpIndex = 0; qfpIndex < qFamilies.size(); ++qfpIndex){
            if ((qFamilies[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) && physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)){
                queueIndex = qfpIndex; break;
            }
        }
        if (queueIndex == ~0) throw std::runtime_error("No queue families found for both graphics and surface support");

        /** 
        auto graphicsQFamilyProperty = std::ranges::find_if(qFamilies, [](auto const &familyProperty){
            return (familyProperty.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
        });
        uint32_t graphicsIndex = std::distance(qFamilies.begin(), graphicsQFamilyProperty);
        */

        float queuePriority = 0.5f;
        vk::DeviceQueueCreateInfo deviceQCreateInfo {.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
        
        //define a PhysicalDeviceFeatures just cuz
        vk::PhysicalDeviceFeatures deviceFeatures;
        vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            //vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
                {.features = {.samplerAnisotropy = true}},
                //{.shaderDrawParameters = true},
                {.synchronization2 = true, .dynamicRendering = true},
                {.extendedDynamicState = true},
            };

        std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName};
        vk::DeviceCreateInfo deviceCreateInfo{
            .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &deviceQCreateInfo,
            .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
            .ppEnabledExtensionNames = requiredDeviceExtension.data()
        };

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);

        graphicsQueue = vk::raii::Queue(device, queueIndex, 0);

    }

    void createSurface(){
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0){
            throw std::runtime_error("Surface creation failed");
        }
        surface = vk::raii::SurfaceKHR(instance, _surface);
    }

    void createSwapchain(){
        auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        swapchainExtent = chooseSwapExtent(surfaceCapabilities);
        uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

        std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
        std::vector<vk::PresentModeKHR> availableModes = physicalDevice.getSurfacePresentModesKHR(*surface);
        swapchainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

        vk::SwapchainCreateInfoKHR swapchainCreateInfo{
            .surface = *surface,
            .minImageCount = minImageCount,
            .imageFormat = swapchainSurfaceFormat.format,
            .imageColorSpace = swapchainSurfaceFormat.colorSpace,
            .imageExtent = swapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = surfaceCapabilities.currentTransform,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = chooseSwapPresentMode(availableModes),
            .clipped = true,
            .oldSwapchain = nullptr
        };

        swapchain = vk::raii::SwapchainKHR(device, swapchainCreateInfo);
        swapchainImages = swapchain.getImages();


    }

    void recreateSwapchain(){
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while ((width == 0) && (height == 0) && !glfwWindowShouldClose(window)){
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        if (glfwWindowShouldClose(window)) return;


        device.waitIdle();
        cleanUpSwapchain();
        createSwapchain();
        createImageViews();
        createDepthResources();
    }

    void cleanUpSwapchain(){
        swapchainImageViews.clear();
        swapchain = nullptr;
    }

    void createGraphicsPipeline(){
        // path is relative to the working directory (usually build/)
        shaderModule = createShaderModule(readFile(COLOR_SHADER_SPV));
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shaderModule, 
            .pName = "vertMain"
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shaderModule,
            .pName = "fragMain"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[2] = {vertShaderStageInfo, fragShaderStageInfo};

        //setup dynamic viewport/scissor so we don't have to compile them each time we want to change it
        std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };
        vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

        //manage how vertices are processed into the pipeline
        auto vertexBindingDescription = Vertex::getBindingDescription();
        auto vertexAttributeDescription = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &vertexBindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescription.size()),
            //.data() returns a pointer to the first element
            .pVertexAttributeDescriptions = vertexAttributeDescription.data()
        };
        

        vk::PipelineInputAssemblyStateCreateInfo assemblyInfo{.topology = vk::PrimitiveTopology::eTriangleList};

        /* NOTE: These are static selections; use if we do not want to use dynamic viewport/scissor
        //setup viewport and scissor
        vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height), 0.0f, 1.0f};
        vk::Rect2D scissor{vk::Offset2D{0,0}, swapchainExtent}
        */

        vk::PipelineRasterizationStateCreateInfo rasterInfo{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False,
            .lineWidth = 1.0f
        };

        //Multisampling/anti-aliasing; default value for now
        vk::PipelineMultisampleStateCreateInfo multisampleInfo{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};


        //Alpha replacement, un-premultiplied blend
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        
        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment
        };

        vk::PipelineLayoutCreateInfo layoutInfo{
            .setLayoutCount = 1, 
            .pSetLayouts = &*descriptorSetLayout, 
            .pushConstantRangeCount = 0
        };

        pipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);


        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = vk::True,
            .depthWriteEnable = vk::True,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineInfochain{ 
            {
            .stageCount = 2,
            .pStages = shaderStages, 
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &assemblyInfo,
            .pViewportState = &viewportState,
            .pDepthStencilState = &depthStencil,
            .pRasterizationState = &rasterInfo,
            .pMultisampleState = &multisampleInfo,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout,
            .renderPass = nullptr
            },

            {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapchainSurfaceFormat.format, .depthAttachmentFormat = findDepthFormat()}
        };

        pipeline = vk::raii::Pipeline(device, nullptr, pipelineInfochain.get<vk::GraphicsPipelineCreateInfo>());

        
        




        
    }
    
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats){
        assert(!availableFormats.empty());
        auto formatIt = std::ranges::find_if(availableFormats,
            [](auto const& availableFormat){ return availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;});
        return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availableModes){
        assert(std::ranges::any_of(availableModes, [](const vk::PresentModeKHR mode){return mode == vk::PresentModeKHR::eFifo;}));
        auto modeIt = std::ranges::find_if(availableModes,
            [](const vk::PresentModeKHR availableMode){ return availableMode == vk::PresentModeKHR::eMailbox;});
        return modeIt != availableModes.end() ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;

    }

    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities){
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) return capabilities.currentExtent;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        return {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities){
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)){
            minImageCount = surfaceCapabilities.maxImageCount;
        }

        return minImageCount;
    }

    vk::raii::ImageView createImageView(vk::Image const &image, vk::Format format, vk::ImageAspectFlags aspectFlags){
        vk::ImageViewCreateInfo viewInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1
            }
        };
        return vk::raii::ImageView(device, viewInfo);

    }

    void createImageViews(){
        assert(swapchainImageViews.empty());
        /** 
        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = swapchainSurfaceFormat.format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        */

        for (auto& image : swapchainImages){
            swapchainImageViews.emplace_back(createImageView(image, swapchainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor));

        }
    }

    void createTextureImageView(){
        textureImageView = createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
    }

    static std::vector<char> readFile(const std::string& filename){
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("failed to open file");

        //tellg : returns where read pointer is ; it should be at the end of the buffer and the size is thus where the read pointer is
        std::vector<char> buffer(file.tellg());
        //set the read pointer to 0
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();
        return buffer;
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{
            .codeSize = code.size() * sizeof(char), 
            .pCode = reinterpret_cast<const uint32_t*>(code.data())
        };

        vk::raii::ShaderModule shaderModule(device, createInfo);
        return shaderModule;
    }

    void createCommandPool(){
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueIndex
        };

        commandPool = vk::raii::CommandPool(device, poolInfo);

    }

    void createCommandBuffers(){
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary, 
            .commandBufferCount = FRAMES_IN_FLIGHT
        };

        //CommandBuffers constructor returns a collection of the commandBuffers, which is natural because the default orientation is multiple
        //frames being drawn to at the same time. 
        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);




    }

    void recordCommandBuffer(uint32_t imageIdx){
        auto &commandBuffer = commandBuffers[frameIdx];

        //beginInfo can depend on what behavior we want, from vk::CommandBufferUsageFlagBits::
        commandBuffer.begin({});

        transitionImageLayout(
            swapchainImages[imageIdx],
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor
            );

        transitionImageLayout(
            *depthImage,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::ImageAspectFlagBits::eDepth
        );

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f,0.0f,0.0f,1.0f);
        vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

        vk::RenderingAttachmentInfo depthAttachmentInfo = {
            .imageView = depthImageView,
            .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .clearValue = clearDepth
        };

        vk::RenderingAttachmentInfo renderingAttachmentInfo{
            .imageView = swapchainImageViews[imageIdx],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor
        };

        vk::RenderingInfo renderingInfo{
            .renderArea = {.offset = {0,0}, .extent = swapchainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &renderingAttachmentInfo,
            .pDepthAttachment = &depthAttachmentInfo
            };

        commandBuffer.beginRendering(renderingInfo);

        //basic drawing commands

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

        commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
        commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value);

        commandBuffer.setViewport(
            0, 
            vk::Viewport(0.0f, static_cast<float>(swapchainExtent.height), static_cast<float>(swapchainExtent.width), -static_cast<float>(swapchainExtent.height), 0.0f, 1.0f)
            );

        commandBuffer.setScissor(
            0,
            vk::Rect2D(vk::Offset2D(0,0), swapchainExtent)
            );

        commandBuffers[frameIdx].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameIdx], nullptr);
        
        //commandBuffer.draw(static_cast<uint32_t>(vertices.size()), 1, 0, 0);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        commandBuffer.endRendering();

        transitionImageLayout(
            swapchainImages[imageIdx],
            vk::ImageLayout::eColorAttachmentOptimal, 
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe,
            vk::ImageAspectFlagBits::eColor
        );

        commandBuffer.end();
        
    }

    /**
     *\brief Image layout transition overload meant for texture image reads/write transitions
    */

    void transitionImageLayout(
        vk::raii::CommandBuffer &commandBuffer,
        vk::raii::Image& image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout
    ){
        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1},
        };

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal){
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal){
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else{
            throw std::runtime_error("Invalid layout transition");
        }

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
    }
    
    /**
    * \brief Image layout transition used for color images and depth buffer images
    */

    void transitionImageLayout(
        vk::Image image,
	    vk::ImageLayout         old_layout,
	    vk::ImageLayout         new_layout,
	    vk::AccessFlags2        src_access_mask,
	    vk::AccessFlags2        dst_access_mask,
	    vk::PipelineStageFlags2 src_stage_mask,
	    vk::PipelineStageFlags2 dst_stage_mask,
        vk::ImageAspectFlags imageAspectFlags
    )
        {
            vk::ImageMemoryBarrier2 barrier = {
                .srcStageMask        = src_stage_mask,
                .srcAccessMask       = src_access_mask,
                .dstStageMask        = dst_stage_mask,
                .dstAccessMask       = dst_access_mask,
                .oldLayout           = old_layout,
                .newLayout           = new_layout,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image               = image,
                .subresourceRange    = {
                    .aspectMask     = imageAspectFlags,
                    .baseMipLevel   = 0,
                    .levelCount     = 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1}
                };
            vk::DependencyInfo dependency_info = {
                .dependencyFlags         = {},
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers    = &barrier
            };
            commandBuffers[frameIdx].pipelineBarrier2(dependency_info);
    }

    void createSyncObjects(){
        assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

        for (int i = 0; i < swapchainImages.size(); ++i){
            presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
        }

        for (int i = 0; i < FRAMES_IN_FLIGHT; ++i){
            renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
            inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }

    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height){
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    static void scrollCallback(GLFWwindow* window, double, double yOffset){
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->applyScroll(static_cast<float>(yOffset));
    }

    /**
    \brief helper to find the memory index of a specified type.
    \param typeFilter bitmask for type that must be supported for the vertex buffer
    \param properties the properties required of the memory for our application
    
    */
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties){
        vk::PhysicalDeviceMemoryProperties memoryProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
        {
            if ((typeFilter) & (1 << i) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }
        throw std::runtime_error("Type not supported in memory properties");

    }

    vk::Format findSupportedFormat(const std::vector<vk::Format> candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features){
        for (const auto format : candidates){
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);
            if (((tiling == vk::ImageTiling::eLinear) && ((props.linearTilingFeatures & features) == features)) ||
            ((tiling == vk::ImageTiling::eOptimal) && ((props.optimalTilingFeatures & features) == features))) {
                return format;
            }

            
        }
        throw std::runtime_error("Failed to find supported format");
    }

    vk::Format findDepthFormat(){
        return findSupportedFormat(
            {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
            vk::ImageTiling::eOptimal,
            vk::FormatFeatureFlagBits::eDepthStencilAttachment
        );
    }

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::Queue graphicsQueue = nullptr;
    uint32_t queueIndex = ~0;
    vk::raii::SwapchainKHR swapchain = nullptr;
    std::vector<vk::Image> swapchainImages;
    vk::SurfaceFormatKHR swapchainSurfaceFormat;
    vk::Extent2D swapchainExtent;
    std::vector<vk::raii::ImageView> swapchainImageViews;
    vk::raii::ShaderModule shaderModule = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;

    std::vector<Vertex> vertices;
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    std::vector<uint32_t> indices;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;

    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    vk::raii::Image textureImage = nullptr;
    vk::raii::DeviceMemory textureImageMemory = nullptr;
    vk::raii::ImageView textureImageView = nullptr;
    vk::raii::Sampler textureSampler = nullptr;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;
    enum class Drag { None, Rotate, Pan, Dolly, Roll };

    uint32_t frameIdx = 0;
    bool framebufferResized = false;
    glm::vec3 eye{50.0f, 2.0f, 2.0f};
    glm::vec3 target{0.0f};
    glm::mat3 camRot{1.0f};
    Drag drag = Drag::None;
    bool leftWasDown = false;
    bool rightWasDown = false;
    double downX = 0.0;
    double downY = 0.0;
    glm::vec3 eyeDown{50.0f, 2.0f, 2.0f};
    glm::vec3 targetDown{0.0f};
    glm::mat3 rotDown{1.0f};
    bool showCooked = false;
    bool spaceWasDown = false;


    GLFWwindow* window = nullptr;
};



int main(){
    try
    {
        HelloTriangleApplication app;
        app.run();
        std::cerr << "hello" << std::endl;
    }

    catch(const std::exception& e){
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}