# steak-sim

Development notes for agents.

## Overview

C++20 + Vulkan project. No databases or external services.

| Component | Purpose |
|-----------|---------|
| `src/sim/` | `steak_voxelizer` library: GPU voxelizer and headless `ComputeContext` |
| `src/render/vulkan_app.cpp` | `steak_sim`: GLFW/Vulkan app that renders and voxelizes the steak |
| `src/render/voxel_viewer.cpp` | `voxel_viewer`: Open3D debug view of the voxel grid |
| `src/shaders/{render,compute}/` | Slang shaders, compiled by `slangc` |
| `tests/` | `voxel_tests` (GoogleTest) plus `voxel_tests.slang` |

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

- All executables are written to `build/`; compiled shaders to `build/shaders/*.spv`.
- Shader and asset paths are compile-time absolute paths (`VOXELIZE_SPV`, `MODEL_SHADER_SPV`, `STEAK_OBJ`, ...), so binaries don't depend on the working directory.
- Requires the Vulkan SDK (`slangc` is found via `$VULKAN_SDK/bin`). The render executables also need GLFW and Open3D (looked up in `~/open3d_install`); pass `-DSTEAK_BUILD_RENDER=OFF` to build only the library and tests.

## Test

```bash
ctest --test-dir build
```

GPU tests skip themselves if no Vulkan 1.3 device with a compute queue is available.

## Notes

- Open3D bundles its own GLFW. Don't link Open3D and Homebrew GLFW into the same executable (macOS warns about duplicate classes and it can crash); `steak_sim` uses only GLFW, `voxel_viewer` only Open3D.
- No linter is configured.
