#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct GridExtent {
  int nx = 0;
  int ny = 0;
  int nz = 0;

  int voxel_count() const { return nx * ny * nz; }
};

struct DiffusionParams {
  float dx = 0.001f;
  float alpha = 1.4e-7f;
  float dt = 0.01f;
};

class HeatGrid {
 public:
  HeatGrid(GridExtent extent, DiffusionParams params);
  ~HeatGrid();

  HeatGrid(const HeatGrid&) = delete;
  HeatGrid& operator=(const HeatGrid&) = delete;

  bool cuda_enabled() const { return cuda_enabled_; }
  const std::string& backend_name() const { return backend_name_; }

  GridExtent extent() const { return extent_; }
  const std::vector<float>& host_data() const { return host_; }

  void fill(float value);
  void set_face_z0(float temperature);
  void step(int iterations = 1);
  void sync_to_host();

 private:
  void host_step();

  GridExtent extent_;
  DiffusionParams params_;
  std::vector<float> host_;
  bool cuda_enabled_ = false;
  std::string backend_name_;
  float* device_grid_ = nullptr;
};
