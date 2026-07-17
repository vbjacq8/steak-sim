cmake_minimum_required(VERSION 3.18)
project(steak-sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(STEAK_BUILD_CUDA "Build CUDA diffusion kernels" ON)

find_package(Vulkan REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

set(SHADER_SOURCES
  ${CMAKE_SOURCE_DIR}/shaders/slice.vert
  ${CMAKE_SOURCE_DIR}/shaders/slice.frag
)
set(SHADER_SPV "")
foreach(shader ${SHADER_SOURCES})
  get_filename_component(shader_name ${shader} NAME)
  set(spv ${CMAKE_BINARY_DIR}/shaders/${shader_name}.spv)
  add_custom_command(
    OUTPUT ${spv}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${GLSLANG_VALIDATOR} -V ${shader} -o ${spv}
    DEPENDS ${shader}
    COMMENT "Compiling shader ${shader_name}"
  )
  list(APPEND SHADER_SPV ${spv})
endforeach()
add_custom_target(shaders DEPENDS ${SHADER_SPV})

set(STEAK_SOURCES
  src/main.cpp
  src/render/vulkan_app.cpp
)

if(STEAK_BUILD_CUDA)
  list(APPEND STEAK_SOURCES src/sim/heat_diffusion.cu)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

add_executable(steak_sim ${STEAK_SOURCES})
add_dependencies(steak_sim shaders)

target_compile_definitions(steak_sim PRIVATE
  STEAK_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
)

target_include_directories(steak_sim PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${Vulkan_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIRS}
)

target_link_directories(steak_sim PRIVATE ${GLFW_LIBRARY_DIRS})

target_link_libraries(steak_sim PRIVATE
  Vulkan::Vulkan
  ${GLFW_LIBRARIES}
)

target_compile_options(steak_sim PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

enable_testing()
add_test(NAME verify COMMAND steak_sim --verify)
