set(VOXELIZE_SRC ${COMPUTE_SHADER_DIR}/voxelize.slang)
set(VOXELIZE_SPV ${APP_SHADER_DIR}/voxelize.spv)
add_custom_command(
  OUTPUT ${VOXELIZE_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${APP_SHADER_DIR}
  COMMAND ${SLANGC_EXECUTABLE} ${VOXELIZE_SRC}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry voxelizeMain
          -o ${VOXELIZE_SPV}
  DEPENDS ${VOXELIZE_SRC} ${COMPUTE_SHADER_DIR}/voxel_math.slang
  COMMENT "Compiling voxelizer compute shader"
  VERBATIM)
add_custom_target(steak_voxelizer_shaders DEPENDS ${VOXELIZE_SPV})
set(STEAK_VOXELIZE_SPV ${VOXELIZE_SPV} PARENT_SCOPE)

add_library(steak_voxelizer STATIC voxelizer.cpp)
add_dependencies(steak_voxelizer steak_voxelizer_shaders)
target_include_directories(steak_voxelizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(steak_voxelizer PUBLIC Vulkan::Vulkan steak_third_party)
target_compile_definitions(steak_voxelizer PUBLIC
  VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
  VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)
