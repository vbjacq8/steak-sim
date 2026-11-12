find_package(glfw3 REQUIRED)
find_package(Open3D REQUIRED HINTS $ENV{HOME}/open3d_install/lib/cmake)

set(MODEL_SHADER_SRC ${RENDER_SHADER_DIR}/28_model_loading.slang)
set(MODEL_SHADER_SPV ${SHADER_OUTPUT_DIR}/model_loading.spv)
add_custom_command(
  OUTPUT ${MODEL_SHADER_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
  COMMAND ${SLANGC_EXECUTABLE} ${MODEL_SHADER_SRC}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry vertMain -entry fragMain
          -o ${MODEL_SHADER_SPV}
  DEPENDS ${MODEL_SHADER_SRC}
  COMMENT "Compiling model shaders for steak_sim"
  VERBATIM)
add_custom_target(steak_sim_shaders DEPENDS ${MODEL_SHADER_SPV})

set(COLOR_SHADER_SRC ${RENDER_SHADER_DIR}/steak_color.slang)
set(COLOR_SHADER_SPV ${SHADER_OUTPUT_DIR}/steak_color.spv)
add_custom_command(
  OUTPUT ${COLOR_SHADER_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
  COMMAND ${SLANGC_EXECUTABLE} ${COLOR_SHADER_SRC} -I ${COMPUTE_SHADER_DIR}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry vertMain -entry fragMain
          -o ${COLOR_SHADER_SPV}
  DEPENDS ${COLOR_SHADER_SRC} ${COMPUTE_SHADER_DIR}/lab_color.slang ${COMPUTE_SHADER_DIR}/cooking_kinetics.slang
  COMMENT "Compiling steak color mesh shader"
  VERBATIM)
add_custom_target(steak_color_shaders DEPENDS ${COLOR_SHADER_SPV})

add_executable(steak_color_demo steak_color_demo.cpp)
add_dependencies(steak_color_demo steak_color_shaders)
target_link_libraries(steak_color_demo PRIVATE steak_voxelizer glfw)
target_compile_definitions(steak_color_demo PRIVATE
  VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
  COLOR_SHADER_SPV="${COLOR_SHADER_SPV}"
  STEAK_OBJ="${STEAK_OBJ}"
  STEAK_TEXTURE="${STEAK_TEXTURE}")

add_executable(steak_sim vulkan_app.cpp)
add_dependencies(steak_sim steak_sim_shaders)
target_link_libraries(steak_sim PRIVATE steak_voxelizer glfw)
target_compile_definitions(steak_sim PRIVATE
  VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
  MODEL_SHADER_SPV="${MODEL_SHADER_SPV}"
  STEAK_OBJ="${STEAK_OBJ}"
  STEAK_TEXTURE="${STEAK_TEXTURE}")

# Standalone so it doesn't load Homebrew's GLFW alongside the copy bundled in Open3D.
add_executable(voxel_viewer voxel_viewer.cpp)
target_link_libraries(voxel_viewer PRIVATE steak_voxelizer Open3D::Open3D)
target_compile_definitions(voxel_viewer PRIVATE
  STEAK_OBJ="${STEAK_OBJ}"
  OPEN3D_RESOURCE_DIR="${Open3D_DIR}/../../../share/resources")

# Heat-map window shared by heat_viewer and steak_heat_demo.
add_library(steak_heat_view STATIC heat_view.cpp)
target_link_libraries(steak_heat_view PUBLIC steak_voxelizer Open3D::Open3D)
target_compile_definitions(steak_heat_view PRIVATE
  OPEN3D_RESOURCE_DIR="${Open3D_DIR}/../../../share/resources")

add_executable(heat_viewer heat_viewer.cpp)
target_link_libraries(heat_viewer PRIVATE steak_heat_view)

add_executable(steak_heat_demo steak_heat_demo.cpp)
target_link_libraries(steak_heat_demo PRIVATE steak_heat_view)
target_compile_definitions(steak_heat_demo PRIVATE
  STEAK_OBJ="${STEAK_OBJ}"
  STEAK_TEXTURE="${STEAK_TEXTURE}")
