cmake_minimum_required(VERSION 3.29)
project(steak-sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(STEAK_BUILD_RENDER "Build the Vulkan steak viewer (src/render)" ON)
option(STEAK_BUILD_TESTS "Build unit tests" ON)

find_package(Vulkan 1.4 REQUIRED)
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

add_library(steak_third_party INTERFACE)
target_include_directories(steak_third_party INTERFACE
  ${glm_SOURCE_DIR}
  ${tinyobjloader_SOURCE_DIR}
  ${stb_SOURCE_DIR})

# Runtime paths in vulkan_app.cpp are relative to the build/ directory.
set(APP_NAME 28_steak_loading)
set(APP_DIR ${CMAKE_BINARY_DIR}/${APP_NAME})
set(APP_SHADER_DIR ${APP_DIR}/shaders)
set(ASSETS_DIR ${CMAKE_SOURCE_DIR}/src/assets)
set(RENDER_SHADER_DIR ${CMAKE_SOURCE_DIR}/src/shaders/render)
set(COMPUTE_SHADER_DIR ${CMAKE_SOURCE_DIR}/src/shaders/compute)

add_subdirectory(src/sim)

if(STEAK_BUILD_RENDER)
  add_subdirectory(src/render)
endif()

if(STEAK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
