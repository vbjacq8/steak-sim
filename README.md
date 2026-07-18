# steak-sim

CUDA-accelerated 3D finite-difference heat diffusion for steak cooking, with Vulkan visualization.

Each voxel stores a temperature. The simulator advances a 3D grid with an explicit finite-difference (FTCS) stencil. Temperature is mapped to color and rendered as a 2D slice today; the target architecture is a full 3D texture / volume ray-march in Vulkan.

## Stack

- **C++17** host code
- **CUDA** for parallel diffusion steps (falls back to CPU when no GPU is present)
- **Vulkan + GLFW** for real-time visualization
- **CMake + Ninja** build system

## Prerequisites (Ubuntu)

```bash
sudo apt-get install -y \
  build-essential ninja-build cmake pkg-config \
  libvulkan-dev vulkan-tools vulkan-validationlayers mesa-vulkan-drivers \
  glslang-tools spirv-tools \
  libglfw3-dev libx11-dev \
  nvidia-cuda-toolkit nvidia-cuda-toolkit-gcc
```

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
```

## Run

Headless verification (diffusion + Vulkan instance check):

```bash
./build/steak_sim --verify
```

Interactive Vulkan slice viewer (requires a display):

```bash
./build/steak_sim --grid 32 --steps 200
```

Options:

| Flag | Description |
|------|-------------|
| `--verify` | Run diffusion + Vulkan checks, then exit |
| `--headless` | Run diffusion only, no window |
| `--grid N` | Cubic grid resolution `N³` (default 32) |
| `--steps N` | Simulation steps / frames (default 100) |

## Roadmap

- [ ] Volume ray-marching shader for full 3D steak texture
- [ ] CUDA–Vulkan interop (export simulation buffer without host round-trip)
- [ ] Steak albedo texture blended with temperature colormap
- [ ] Boundary conditions for pan contact and ambient cooling
