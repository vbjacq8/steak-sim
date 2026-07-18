#include "sim/heat_diffusion.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace {

__global__ void diffuse_kernel(float* grid, int nx, int ny, int nz, float alpha,
                               float dx, float dt) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= nx || y >= ny || z >= nz) {
    return;
  }

  const int idx = x + nx * (y + ny * z);
  const float center = grid[idx];

  const auto at = [&](int ix, int iy, int iz) -> float {
    ix = max(0, min(nx - 1, ix));
    iy = max(0, min(ny - 1, iy));
    iz = max(0, min(nz - 1, iz));
    return grid[ix + nx * (iy + ny * iz)];
  };

  const float lap =
      (at(x + 1, y, z) + at(x - 1, y, z) + at(x, y + 1, z) + at(x, y - 1, z) +
       at(x, y, z + 1) + at(x, y, z - 1) - 6.0f * center) /
      (dx * dx);

  grid[idx] = center + alpha * dt * lap;
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

}  // namespace

HeatGrid::HeatGrid(GridExtent extent, DiffusionParams params)
    : extent_(extent), params_(params), host_(extent.voxel_count(), 0.0f) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    backend_name_ = "CPU (no CUDA device)";
    return;
  }

  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  const std::size_t bytes = host_.size() * sizeof(float);
  check_cuda(cudaMalloc(&device_grid_, bytes), "cudaMalloc");
  check_cuda(cudaMemcpy(device_grid_, host_.data(), bytes, cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");

  cuda_enabled_ = true;
  backend_name_ = std::string("CUDA: ") + prop.name;
}

HeatGrid::~HeatGrid() {
  if (device_grid_ != nullptr) {
    cudaFree(device_grid_);
    device_grid_ = nullptr;
  }
}

void HeatGrid::fill(float value) {
  std::fill(host_.begin(), host_.end(), value);
  if (cuda_enabled_) {
  const std::size_t bytes = host_.size() * sizeof(float);
    check_cuda(cudaMemcpy(device_grid_, host_.data(), bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy fill");
  }
}

void HeatGrid::set_face_z0(float temperature) {
  for (int y = 0; y < extent_.ny; ++y) {
    for (int x = 0; x < extent_.nx; ++x) {
      host_[x + extent_.nx * y] = temperature;
    }
  }
  if (cuda_enabled_) {
    const std::size_t row_bytes = extent_.nx * sizeof(float);
    for (int y = 0; y < extent_.ny; ++y) {
      float* row = device_grid_ + extent_.nx * y;
      check_cuda(cudaMemcpy(row, &host_[extent_.nx * y], row_bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy boundary");
    }
  }
}

void HeatGrid::host_step() {
  std::vector<float> next = host_;
  const int nx = extent_.nx;
  const int ny = extent_.ny;
  const int nz = extent_.nz;
  const float inv_dx2 = 1.0f / (params_.dx * params_.dx);

  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const int idx = x + nx * (y + ny * z);
        const float center = host_[idx];
        const auto at = [&](int ix, int iy, int iz) {
          ix = std::max(0, std::min(nx - 1, ix));
          iy = std::max(0, std::min(ny - 1, iy));
          iz = std::max(0, std::min(nz - 1, iz));
          return host_[ix + nx * (iy + ny * iz)];
        };
        const float lap = (at(x + 1, y, z) + at(x - 1, y, z) + at(x, y + 1, z) +
                           at(x, y - 1, z) + at(x, y, z + 1) + at(x, y, z - 1) -
                           6.0f * center) *
                          inv_dx2;
        next[idx] = center + params_.alpha * params_.dt * lap;
      }
    }
  }
  host_.swap(next);
}

void HeatGrid::step(int iterations) {
  for (int i = 0; i < iterations; ++i) {
    set_face_z0(200.0f);
    if (!cuda_enabled_) {
      host_step();
      continue;
    }

    const dim3 block(8, 8, 8);
    const dim3 grid((extent_.nx + block.x - 1) / block.x,
                    (extent_.ny + block.y - 1) / block.y,
                    (extent_.nz + block.z - 1) / block.z);
    diffuse_kernel<<<grid, block>>>(device_grid_, extent_.nx, extent_.ny, extent_.nz,
                                    params_.alpha, params_.dx, params_.dt);
    check_cuda(cudaGetLastError(), "diffuse_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }
}

void HeatGrid::sync_to_host() {
  if (!cuda_enabled_) {
    return;
  }
  const std::size_t bytes = host_.size() * sizeof(float);
  check_cuda(cudaMemcpy(host_.data(), device_grid_, bytes, cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H");
}
