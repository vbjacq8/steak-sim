# steak-sim

Development notes for Cursor Cloud agents.

## Overview

Single C++/CUDA/Vulkan executable (`steak_sim`). No databases or external services.

| Component | Purpose |
|-----------|---------|
| `steak_sim` | Host binary: simulation loop + Vulkan slice viewer |
| CUDA kernels (`src/sim/heat_diffusion.cu`) | 3D FTCS diffusion on GPU |
| Vulkan renderer (`src/render/vulkan_app.cpp`) | Temperature slice visualization |

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build -j
```

Use `g++` as the C++ compiler. `/usr/bin/c++` may point at Clang without a working `libstdc++` link on this image.

Shaders compile automatically via `glslangValidator` into `build/shaders/*.spv`.

## Test / verify

```bash
./build/steak_sim --verify
ctest --test-dir build
```

`--verify` runs CPU or CUDA diffusion (whichever is available) and checks that Vulkan can create an instance and enumerate devices.

## Run interactive viewer

Requires `DISPLAY` (Cloud Agent VMs expose `:1`) and `XDG_RUNTIME_DIR`:

```bash
export DISPLAY=:1
export XDG_RUNTIME_DIR=/tmp/xdg-runtime
mkdir -p "$XDG_RUNTIME_DIR"
./build/steak_sim --grid 32 --steps 100
```

## CUDA / GPU notes

- **Compile**: `nvidia-cuda-toolkit` provides `nvcc` even without a GPU.
- **Runtime**: CUDA kernels need an NVIDIA GPU. Cloud Agent VMs typically have **no GPU**; the code falls back to a CPU reference implementation automatically.
- For local GPU development, ensure `nvidia-smi` reports a device; the binary will print `CUDA: <device name>` at startup.

## Vulkan notes

- Software rendering is available via Mesa Lavapipe (`lvp_icd.json`) if no discrete GPU is present.
- `vulkaninfo` may fail on some headless setups even when GLFW + Vulkan rendering works through `DISPLAY=:1`.

## Lint

No project linter is configured yet. Build with `-Wall -Wextra -Wpedantic` is enabled for C++ sources in `CMakeLists.txt`.
