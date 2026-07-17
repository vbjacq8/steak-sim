#include "render/vulkan_app.h"
#include "sim/heat_diffusion.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

namespace {

struct Options {
  int nx = 32;
  int ny = 32;
  int nz = 32;
  int steps = 100;
  bool verify = false;
  bool headless = false;
};

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--verify") == 0) {
      opts.verify = true;
    } else if (std::strcmp(argv[i], "--headless") == 0) {
      opts.headless = true;
    } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
      opts.steps = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
      const int n = std::stoi(argv[++i]);
      opts.nx = opts.ny = opts.nz = n;
    }
  }
  return opts;
}

float grid_min(const std::vector<float>& grid) {
  return *std::min_element(grid.begin(), grid.end());
}

float grid_max(const std::vector<float>& grid) {
  return *std::max_element(grid.begin(), grid.end());
}

int run_verify(const Options& opts) {
  std::cout << "steak-sim verify\n";

  const GridExtent extent{opts.nx, opts.ny, opts.nz};
  DiffusionParams params{};
  params.dx = 0.01f / static_cast<float>(opts.nx);
  params.dt = 0.2f * params.dx * params.dx / params.alpha;

  HeatGrid grid(extent, params);
  grid.fill(4.0f);
  grid.set_face_z0(200.0f);

  const auto t0 = std::chrono::steady_clock::now();
  grid.step(opts.steps);
  grid.sync_to_host();
  const auto ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();

  const float center = grid.host_data()[extent.nx / 2 + extent.nx * (extent.ny / 2 +
                                                                     extent.ny * (extent.nz / 2))];
  std::cout << "diffusion backend: " << grid.backend_name() << '\n';
  std::cout << "grid: " << opts.nx << 'x' << opts.ny << 'x' << opts.nz << " steps=" << opts.steps
            << " elapsed=" << ms << "ms\n";
  std::cout << "temperature min=" << grid_min(grid.host_data())
            << " max=" << grid_max(grid.host_data()) << " center=" << center << '\n';

  const bool diffusion_ok = center > 4.0f && grid_max(grid.host_data()) <= 200.01f;
  const bool vulkan_ok = verify_vulkan_instance();
  std::cout << "vulkan: " << (vulkan_ok ? "ok" : "failed") << '\n';
  std::cout << "RESULT: " << (diffusion_ok && vulkan_ok ? "PASS" : "FAIL") << '\n';
  return (diffusion_ok && vulkan_ok) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = parse_args(argc, argv);
  if (opts.verify) {
    return run_verify(opts);
  }

  const GridExtent extent{opts.nx, opts.ny, opts.nz};
  DiffusionParams params{};
  params.dx = 0.01f / static_cast<float>(opts.nx);
  params.dt = 0.2f * params.dx * params.dx / params.alpha;

  HeatGrid grid(extent, params);
  grid.fill(4.0f);
  std::cout << "backend: " << grid.backend_name() << '\n';

  if (opts.headless) {
    grid.step(opts.steps);
    grid.sync_to_host();
    std::cout << "headless run complete. center="
              << grid.host_data()[extent.nx / 2 + extent.nx * (extent.ny / 2)] << "C\n";
    return 0;
  }

  VulkanSliceViewer viewer(960, 720, "steak-sim");
  if (!viewer.ok()) {
    std::cerr << "Vulkan viewer failed: " << viewer.error() << '\n';
    return 1;
  }

  int frame = 0;
  while (true) {
    grid.step(2);
    grid.sync_to_host();
    const auto slice = make_temperature_slice(grid.host_data(), extent.nx, extent.ny, extent.nz,
                                              0, 4.0f, 200.0f);
    if (!viewer.pump_once(slice)) {
      break;
    }
    if (++frame >= opts.steps) {
      break;
    }
  }

  std::cout << "rendered " << frame << " frames\n";
  return 0;
}
