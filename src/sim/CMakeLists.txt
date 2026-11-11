set(VOXELIZE_SRC ${COMPUTE_SHADER_DIR}/voxelize.slang)
set(VOXELIZE_SPV ${SHADER_OUTPUT_DIR}/voxelize.spv)
add_custom_command(
  OUTPUT ${VOXELIZE_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
  COMMAND ${SLANGC_EXECUTABLE} ${VOXELIZE_SRC}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry voxelizeMain
          -o ${VOXELIZE_SPV}
  DEPENDS ${VOXELIZE_SRC} ${COMPUTE_SHADER_DIR}/voxel_math.slang
  COMMENT "Compiling voxelizer compute shader"
  VERBATIM)

set(FINITE_DIFFERENCE_SRC ${COMPUTE_SHADER_DIR}/finitedifference.slang)
set(FINITE_DIFFERENCE_SPV ${SHADER_OUTPUT_DIR}/finitedifference.spv)
add_custom_command(
  OUTPUT ${FINITE_DIFFERENCE_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
  COMMAND ${SLANGC_EXECUTABLE} ${FINITE_DIFFERENCE_SRC}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry heatStepMain
          -o ${FINITE_DIFFERENCE_SPV}
  DEPENDS ${FINITE_DIFFERENCE_SRC}
  COMMENT "Compiling heat equation compute shader"
  VERBATIM)
set(COLOR_INIT_SRC ${COMPUTE_SHADER_DIR}/color_init.slang)
set(COLOR_INIT_SPV ${SHADER_OUTPUT_DIR}/color_init.spv)
add_custom_command(
  OUTPUT ${COLOR_INIT_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
  COMMAND ${SLANGC_EXECUTABLE} ${COLOR_INIT_SRC}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry colorInitMain
          -o ${COLOR_INIT_SPV}
  DEPENDS ${COLOR_INIT_SRC} ${COMPUTE_SHADER_DIR}/lab_color.slang
  COMMENT "Compiling Lab texture init shader"
  VERBATIM)

set(COLOR_KINETICS_SRC ${COMPUTE_SHADER_DIR}/color_kinetics.slang)
set(COLOR_KINETICS_SPV ${SHADER_OUTPUT_DIR}/color_kinetics.spv)
add_custom_command(
  OUTPUT ${COLOR_KINETICS_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
  COMMAND ${SLANGC_EXECUTABLE} ${COLOR_KINETICS_SRC}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry colorKineticsMain
          -o ${COLOR_KINETICS_SPV}
  DEPENDS ${COLOR_KINETICS_SRC}
  COMMENT "Compiling color kinetics shader"
  VERBATIM)

set(COLOR_EXPORT_SRC ${COMPUTE_SHADER_DIR}/color_export.slang)
set(COLOR_EXPORT_SPV ${SHADER_OUTPUT_DIR}/color_export.spv)
add_custom_command(
  OUTPUT ${COLOR_EXPORT_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
  COMMAND ${SLANGC_EXECUTABLE} ${COLOR_EXPORT_SRC}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry colorExportMain
          -o ${COLOR_EXPORT_SPV}
  DEPENDS ${COLOR_EXPORT_SRC} ${COMPUTE_SHADER_DIR}/lab_color.slang
  COMMENT "Compiling Lab display shader"
  VERBATIM)

add_custom_target(steak_voxelizer_shaders DEPENDS ${VOXELIZE_SPV} ${FINITE_DIFFERENCE_SPV}
                  ${COLOR_INIT_SPV} ${COLOR_KINETICS_SPV} ${COLOR_EXPORT_SPV})

add_library(steak_voxelizer STATIC mesh_loader.cpp voxelizer.cpp compute_context.cpp heat_solver.cpp steak_pipeline.cpp color_solver.cpp)
add_dependencies(steak_voxelizer steak_voxelizer_shaders)
target_include_directories(steak_voxelizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(steak_voxelizer PUBLIC Vulkan::Vulkan steak_third_party)
target_compile_definitions(steak_voxelizer PUBLIC
  VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
  VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  VOXELIZE_SPV="${VOXELIZE_SPV}"
  FINITE_DIFFERENCE_SPV="${FINITE_DIFFERENCE_SPV}"
  COLOR_INIT_SPV="${COLOR_INIT_SPV}"
  COLOR_KINETICS_SPV="${COLOR_KINETICS_SPV}"
  COLOR_EXPORT_SPV="${COLOR_EXPORT_SPV}")
