cmake_minimum_required(VERSION 3.29)
project(steak_render LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan 1.4 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Open3D REQUIRED HINTS $ENV{HOME}/open3d_install/lib/cmake)
find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

# Header-only deps; SOURCE_SUBDIR points at a nonexistent dir so their own
# CMakeLists aren't added (vulkan_app.cpp defines the *_IMPLEMENTATION macros).
include(FetchContent)
FetchContent_Declare(glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR none)
FetchContent_Declare(tinyobjloader
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG v2.0.0rc13
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR none)
FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG master
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR none)
FetchContent_MakeAvailable(glm tinyobjloader stb)

# Runtime paths in vulkan_app.cpp are relative to the build/ directory.
set(APP_NAME 28_steak_loading)
set(APP_DIR ${CMAKE_BINARY_DIR}/${APP_NAME})
set(ASSETS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../assets)

set(SLANG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/28_model_loading.slang)
set(SLANG_SPV ${APP_DIR}/shaders/slang.spv)
add_custom_command(
  OUTPUT ${SLANG_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${APP_DIR}/shaders
  COMMAND ${SLANGC_EXECUTABLE} ${SLANG_SRC}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry vertMain -entry fragMain
          -o ${SLANG_SPV}
  DEPENDS ${SLANG_SRC}
  COMMENT "Compiling Slang shaders for ${APP_NAME}"
  VERBATIM)
add_custom_target(${APP_NAME}_shaders DEPENDS ${SLANG_SPV})

add_executable(${APP_NAME} vulkan_app.cpp)
add_dependencies(${APP_NAME} ${APP_NAME}_shaders)
set_target_properties(${APP_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${APP_DIR})

target_include_directories(${APP_NAME} PRIVATE
  ${glm_SOURCE_DIR}
  ${tinyobjloader_SOURCE_DIR}
  ${stb_SOURCE_DIR})

target_link_libraries(${APP_NAME} PRIVATE Vulkan::Vulkan glfw Open3D::Open3D)

target_compile_definitions(${APP_NAME} PRIVATE
  VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
  VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS)

file(COPY ${ASSETS_DIR}/models/steak.obj DESTINATION ${APP_DIR}/models)
file(COPY ${ASSETS_DIR}/textures/textureSteak.jpeg DESTINATION ${APP_DIR}/textures)
