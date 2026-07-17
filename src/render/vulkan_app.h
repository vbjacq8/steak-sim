#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct SliceImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgba;
};

class VulkanSliceViewer {
 public:
  VulkanSliceViewer(int width, int height, const char* title);
  ~VulkanSliceViewer();

  VulkanSliceViewer(const VulkanSliceViewer&) = delete;
  VulkanSliceViewer& operator=(const VulkanSliceViewer&) = delete;

  bool ok() const { return ok_; }
  const std::string& error() const { return error_; }

  bool pump_once(const SliceImage& slice);

 private:
  struct Impl;
  Impl* impl_ = nullptr;

  bool ok_ = false;
  std::string error_;
};

bool verify_vulkan_instance();

SliceImage make_temperature_slice(const std::vector<float>& grid, int nx, int ny, int nz,
                                  int z_layer, float min_temp, float max_temp);
