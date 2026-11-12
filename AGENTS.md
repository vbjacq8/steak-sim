# steak-sim

Development notes for agents.

## Overview

C++20 + Vulkan project. No databases or external services.

| Component | Purpose |
|-----------|---------|
| `src/sim/` | `steak_voxelizer` library: GPU voxelizer, finite-difference `HeatSolver` (and `HeatSession`, which keeps a field on the GPU between batches of steps), `simulateMeshHeat` pipeline (mesh -> voxels -> heat) with its live counterpart `MeshHeatColorSession`, and headless `ComputeContext` |
| `src/render/vulkan_app.cpp` | `steak_sim`: GLFW/Vulkan app that renders and voxelizes the steak |
| `src/render/voxel_viewer.cpp` | `voxel_viewer`: Open3D debug view of the voxel grid |
| `src/render/heat_view.cpp` | `steak_heat_view` library: Open3D heat-map window shared by the heat executables |
| `src/render/heat_viewer.cpp` | `heat_viewer`: heat sim on a solid block, shown in the heat-map window |
| `src/render/steak_heat_demo.cpp` | `steak_heat_demo`: voxel heat and cooked color; toggles the temperature map, that texture, and the textured mesh |
| `src/render/steak_heat_live.cpp` | `steak_heat_live`: the same cook played on a clock (Start/Pause/Reset, adjustable speed) in the same window; the heat-map window's live mode drives it from a worker thread |
| `src/render/demo_mesh.cpp` | `steak_demo_mesh`: model loading, texture-atlas packing and scaling shared by the two steak demos |
| `src/tools/doneness_calibrate.cpp` | `doneness_calibrate`: headless; cooks a flipped 1D column to a center temperature and prints each layer's peak, for calibrating the doneness table |
| `src/shaders/{render,compute}/` | Slang shaders, compiled by `slangc` |
| `tests/` | `voxel_tests` and `heat_tests` (GoogleTest) plus `voxel_tests.slang` |

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

- All executables are written to `build/`; compiled shaders to `build/shaders/*.spv`.
- Shader and asset paths are compile-time absolute paths (`VOXELIZE_SPV`, `MODEL_SHADER_SPV`, `STEAK_OBJ`, ...), so binaries don't depend on the working directory.
- Requires the Vulkan SDK (`slangc` is found via `$VULKAN_SDK/bin`). The render executables also need GLFW and Open3D (looked up in `~/open3d_install`); pass `-DSTEAK_BUILD_RENDER=OFF` to build only the library and tests.
- The heat sim scenario is the `HeatParams` defaults in `src/sim/heat_solver.hpp`. `steak_heat_demo` treats the mesh as centimeters. The explicit scheme needs `alpha*dt/dx^2 <= ~1/6`.
- The pan is the low layer along `HeatParams::panAxis` (z by default). The steak demos set it to the model's thinnest axis: y for steak1, z for steak2.

## Test

```bash
ctest --test-dir build
```

GPU tests skip themselves if no Vulkan 1.3 device with a compute queue is available.

## Notes

- Open3D bundles its own GLFW. Don't link Open3D and Homebrew GLFW into the same executable (macOS warns about duplicate classes and it can crash); `steak_sim` uses only GLFW, `voxel_viewer`, `heat_viewer` and `steak_heat_demo` only Open3D.
- Open3D 0.20 caches textures by a hash of their pixels and does not free a texture when its geometry is removed. The heat window gives every surface texture unique pixels and releases it itself (`uniqueTexture` and `removeSurface` in `heat_view.cpp`); keep that when changing how the mesh is drawn.
- No linter is configured.
