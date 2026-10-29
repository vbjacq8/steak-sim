find_package(glfw3 REQUIRED)
find_package(Open3D REQUIRED HINTS $ENV{HOME}/open3d_install/lib/cmake)

set(SLANG_SRC ${RENDER_SHADER_DIR}/28_model_loading.slang)
set(SLANG_SPV ${APP_SHADER_DIR}/slang.spv)
add_custom_command(
  OUTPUT ${SLANG_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${APP_SHADER_DIR}
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

target_link_libraries(${APP_NAME} PRIVATE steak_voxelizer glfw Open3D::Open3D)

target_compile_definitions(${APP_NAME} PRIVATE
  VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS)

# Standalone so it doesn't load Homebrew's GLFW alongside the copy bundled in Open3D.
add_executable(voxel_viewer voxel_viewer.cpp)
target_link_libraries(voxel_viewer PRIVATE steak_voxelizer Open3D::Open3D)
target_compile_definitions(voxel_viewer PRIVATE
  VOXELIZE_SPV="${STEAK_VOXELIZE_SPV}"
  STEAK_OBJ="${ASSETS_DIR}/models/steak.obj"
  OPEN3D_RESOURCE_DIR="${Open3D_DIR}/../../../share/resources")

file(COPY ${ASSETS_DIR}/models/steak.obj DESTINATION ${APP_DIR}/models)
file(COPY ${ASSETS_DIR}/textures/textureSteak.jpeg DESTINATION ${APP_DIR}/textures)
