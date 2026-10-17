include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
  GIT_SHALLOW TRUE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/voxel_tests.slang)
set(TEST_SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shaders/voxel_tests.spv)
add_custom_command(
  OUTPUT ${TEST_SHADER_SPV}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
  COMMAND ${SLANGC_EXECUTABLE} ${TEST_SHADER_SRC} -I ${COMPUTE_SHADER_DIR}
          -target spirv -profile spirv_1_4 -emit-spirv-directly
          -fvk-use-entrypoint-name -entry testMain
          -o ${TEST_SHADER_SPV}
  DEPENDS ${TEST_SHADER_SRC} ${COMPUTE_SHADER_DIR}/voxel_math.slang
  COMMENT "Compiling voxel test shader"
  VERBATIM)
add_custom_target(voxel_test_shaders DEPENDS ${TEST_SHADER_SPV})

add_executable(voxel_tests voxel_tests.cpp)
add_dependencies(voxel_tests voxel_test_shaders)
target_link_libraries(voxel_tests PRIVATE steak_voxelizer GTest::gtest_main)
target_compile_definitions(voxel_tests PRIVATE
  VOXELIZE_SPV="${STEAK_VOXELIZE_SPV}"
  VOXEL_TESTS_SPV="${TEST_SHADER_SPV}"
  STEAK_OBJ="${ASSETS_DIR}/models/steak.obj")

include(GoogleTest)
gtest_discover_tests(voxel_tests DISCOVERY_TIMEOUT 60)
