// Heat-map window shared by heat_viewer and steak_heat_demo. Only the faces between a drawn cell and
// an undrawn one (air, hidden pan, outside the slice, or outside the grid) are built, so a 128^3 grid
// stays at ~100k quads.
//
// The side panel keeps a temperature key, then a reset button, then collapsible settings. Typing in a
// box stores that value; the blocks change when a reset finishes.

#include "heat_view.hpp"

#include "color_model.hpp"
#include "reset_dialog.hpp"
#include "voxelizer.hpp"


#include <open3d/Open3D.h>
#include <open3d/visualization/rendering/ColorGrading.h>
#include <open3d/visualization/rendering/View.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace gui = open3d::visualization::gui;
namespace rendering = open3d::visualization::rendering;

namespace {

constexpr double BLOCK_SHRINK = 0.9;
constexpr double MM_PER_M = 1000.0;
constexpr int LEGEND_ROWS = 7;
constexpr float COLOR_MIN_F = 70.0f;
constexpr float COLOR_MAX_F = 200.0f;
constexpr int MAX_DIM = 256;
constexpr long MAX_STEPS = 100000;
constexpr std::array<const char*, 3> AXIS_NAMES{"X", "Y", "Z"};

std::string format(const char* fmt, double value){
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), fmt, value);
    return buffer;
}

bool parseNumber(const char* text, double& value, std::string& error){
    if (!text){
        error = "Enter a number.";
        return false;
    }
    while (*text == ' ' || *text == '\t') ++text;
    if (*text == '\0'){
        error = "Enter a number.";
        return false;
    }
    char* end = nullptr;
    value = std::strtod(text, &end);
    if (end == text){
        error = "Enter a number.";
        return false;
    }
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0' || !std::isfinite(value)){
        error = "Enter a number.";
        return false;
    }
    return true;
}

bool parseCount(const char* text, long& value, long min, long max, std::string& error, const char* what){
    double number = 0.0;
    if (!parseNumber(text, number, error)) return false;
    if (number < min || number > max || std::abs(number - std::round(number)) > 1e-6){
        error = std::string(what) + " must be a whole number from " + std::to_string(min) + " to " + std::to_string(max) + ".";
        return false;
    }
    value = std::lround(number);
    return true;
}

Eigen::Vector3d heatColor(double t){
    static constexpr std::array<std::array<double, 3>, 5> stops{{
        {0.12, 0.25, 0.85}, {0.10, 0.75, 0.90}, {0.98, 0.88, 0.15}, {0.98, 0.45, 0.05}, {0.75, 0.05, 0.05}}};
    const double x = std::clamp(t, 0.0, 1.0) * (stops.size() - 1);
    const size_t i = std::min(static_cast<size_t>(x), stops.size() - 2);
    const double f = x - i;
    return {stops[i][0] + f * (stops[i + 1][0] - stops[i][0]), stops[i][1] + f * (stops[i + 1][1] - stops[i][1]),
            stops[i][2] + f * (stops[i + 1][2] - stops[i][2])};
}

std::shared_ptr<gui::ImageWidget> makeSwatch(const Eigen::Vector3d& color, int width, int height){
    auto image = std::make_shared<open3d::geometry::Image>();
    image->Prepare(width, height, 3, 1);
    for (int i = 0; i < width * height; ++i){
        for (int c = 0; c < 3; ++c) image->data_[3 * i + c] = static_cast<uint8_t>(std::lround(255.0 * color[c]));
    }
    return std::make_shared<gui::ImageWidget>(image);
}

void appendFace(open3d::geometry::TriangleMesh& blocks, const HeatGrid& grid, glm::uvec3 cell, int axis, bool positive,
                float coldTemp, float hotTemp, bool showCookedColor, bool showInitialColor){
    const int u = (axis + 1) % 3;
    const int v = (axis + 2) % 3;
    const double s = grid.spacing * MM_PER_M;
    const double inset = 0.5 * (1.0 - BLOCK_SHRINK) * s;
    Eigen::Vector3d lo, hi;
    for (int a = 0; a < 3; ++a){
        lo[a] = cell[a] * s + inset;
        hi[a] = (cell[a] + 1) * s - inset;
    }
    const double plane = positive ? hi[axis] : lo[axis];
    const std::array<std::array<double, 2>, 4> corners{{{lo[u], lo[v]}, {hi[u], lo[v]}, {hi[u], hi[v]}, {lo[u], hi[v]}}};

    const int base = static_cast<int>(blocks.vertices_.size());
    const bool solid = grid.kind(cell.x, cell.y, cell.z) == CELL_SOLID;
    const bool initial = showInitialColor && solid && grid.initialColor.size() == grid.cellCount();
    const bool meat = !initial && showCookedColor && solid && grid.color.size() == grid.cellCount();
    Eigen::Vector3d color;
    if (initial || meat){
        // Already linear sRGB from the Lab export. The heat-map stops are display values and get
        // linearized with pow(2.2); doing that again would darken the steak.
        const glm::vec3 sample = (initial ? grid.initialColor : grid.color)[grid.index(cell.x, cell.y, cell.z)];
        auto channel = [](float value){ return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f; };
        color = {channel(sample.x), channel(sample.y), channel(sample.z)};
    } else{
        const double t = hotTemp > coldTemp ? (grid.at(cell.x, cell.y, cell.z) - coldTemp) / (hotTemp - coldTemp) : 1.0;
        color = heatColor(t).array().pow(2.2);
    }
    for (const auto& [cu, cv] : corners){
        Eigen::Vector3d p;
        p[axis] = plane;
        p[u] = cu;
        p[v] = cv;
        blocks.vertices_.push_back(p);
        blocks.vertex_colors_.push_back(color);
    }
    if (positive){
        blocks.triangles_.emplace_back(base, base + 1, base + 2);
        blocks.triangles_.emplace_back(base, base + 2, base + 3);
    } else{
        blocks.triangles_.emplace_back(base, base + 2, base + 1);
        blocks.triangles_.emplace_back(base, base + 3, base + 2);
    }
}

// Builds the block mesh for the current slice. Safe to call off the UI thread. `progress` is the fraction of
// z-slices finished.
open3d::geometry::TriangleMesh buildHeatMesh(const HeatGrid& grid, int sliceAxis, int slice, bool showPan, float coldTemp,
                                             float hotTemp, bool showCookedColor, bool showInitialColor,
                                             const std::function<void(float)>& progress){
    glm::uvec3 regionLo(0);
    glm::uvec3 regionHi = grid.dims - 1u;
    if (slice >= 0){
        const auto clamped = static_cast<uint32_t>(std::min(slice, static_cast<int>(grid.dims[sliceAxis]) - 1));
        regionLo[sliceAxis] = regionHi[sliceAxis] = clamped;
    }

    auto drawn = [&](glm::ivec3 cell){
        if (!glm::all(glm::greaterThanEqual(cell, glm::ivec3(regionLo))) || !glm::all(glm::lessThanEqual(cell, glm::ivec3(regionHi))))
            return false;
        const uint32_t kind = grid.kind(cell.x, cell.y, cell.z);
        return kind == CELL_SOLID || (kind == CELL_PAN && showPan);
    };

    open3d::geometry::TriangleMesh blocks;
    const uint32_t zSpan = regionHi.z - regionLo.z + 1u;
    uint32_t zDone = 0;
    for (uint32_t z = regionLo.z; z <= regionHi.z; ++z){
        for (uint32_t y = regionLo.y; y <= regionHi.y; ++y){
            for (uint32_t x = regionLo.x; x <= regionHi.x; ++x){
                const glm::ivec3 cell(x, y, z);
                if (!drawn(cell)) continue;
                for (int axis = 0; axis < 3; ++axis){
                    for (int side : {-1, 1}){
                        glm::ivec3 neighbor = cell;
                        neighbor[axis] += side;
                        if (!drawn(neighbor)){
                            appendFace(blocks, grid, glm::uvec3(cell), axis, side > 0, coldTemp, hotTemp, showCookedColor, showInitialColor);
                        }
                    }
                }
            }
        }
        if (progress) progress(static_cast<float>(++zDone) / static_cast<float>(zSpan));
    }
    return blocks;
}

bool hasSurface(const HeatSurface& surface){
    return surface.metersPerUnit > 0.0f && !surface.positions.empty() && surface.indices.size() >= 3;
}

struct GridFrame{
    glm::vec3 origin{};
    float voxelSize = 1.0f;
    float midZ = 0.0f;  // model-space height halfway up the mesh; splits the pan side from the top
};

// Same corner the voxelizer uses, so a model-space point lands in the cell the blocks draw for that index.
GridFrame gridFrame(const HeatGrid& grid, const HeatSurface& surface){
    glm::vec3 low(std::numeric_limits<float>::infinity());
    glm::vec3 high(-std::numeric_limits<float>::infinity());
    for (const glm::vec3& point : surface.positions){
        low = glm::min(low, point);
        high = glm::max(high, point);
    }
    const VoxelGrid layout = makeVoxelGridLayout(low, high, grid.spacing / surface.metersPerUnit);
    return {layout.origin, layout.voxelSize, 0.5f * (low.z + high.z)};
}

Eigen::Vector3d sampleVoxel(const HeatGrid& grid, const GridFrame& frame, glm::vec3 position, float coldTemp, float hotTemp,
                            bool showCookedColor){
    const glm::vec3 coord = (position - frame.origin) / frame.voxelSize - 0.5f;
    const int x0 = static_cast<int>(std::floor(coord.x));
    const int y0 = static_cast<int>(std::floor(coord.y));
    const int z0 = static_cast<int>(std::floor(coord.z));
    const float tx = coord.x - x0;
    const float ty = coord.y - y0;
    const float tz = coord.z - z0;
    const bool cooked = showCookedColor && grid.color.size() == grid.cellCount();
    float weight = 0.0f;
    float temperature = 0.0f;
    glm::vec3 color(0.0f);
    for (int dz = 0; dz < 2; ++dz){
        for (int dy = 0; dy < 2; ++dy){
            for (int dx = 0; dx < 2; ++dx){
                const int x = x0 + dx;
                const int y = y0 + dy;
                const int z = z0 + dz;
                if (x < 0 || y < 0 || z < 0 || x >= static_cast<int>(grid.dims.x) || y >= static_cast<int>(grid.dims.y) ||
                    z >= static_cast<int>(grid.dims.z)){
                    continue;
                }
                if (grid.kind(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)) != CELL_SOLID) continue;
                const float w = (dx ? tx : 1.0f - tx) * (dy ? ty : 1.0f - ty) * (dz ? tz : 1.0f - tz);
                weight += w;
                temperature += w * grid.at(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z));
                if (cooked) color += w * grid.color[grid.index(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z))];
            }
        }
    }
    if (!(weight > 1.0e-6f)){
        const int x = std::clamp(static_cast<int>(std::lround(coord.x)), 0, static_cast<int>(grid.dims.x) - 1);
        const int y = std::clamp(static_cast<int>(std::lround(coord.y)), 0, static_cast<int>(grid.dims.y) - 1);
        const int z = std::clamp(static_cast<int>(std::lround(coord.z)), 0, static_cast<int>(grid.dims.z) - 1);
        temperature = grid.at(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z));
        if (cooked && grid.kind(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z)) == CELL_SOLID){
            color = grid.color[grid.index(static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z))];
            weight = 1.0f;
        }
    } else if (cooked){
        color /= weight;
        temperature /= weight;
    } else{
        temperature /= weight;
    }
    if (cooked && weight > 1.0e-6f){
        auto channel = [](float value){ return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f; };
        return {channel(color.x), channel(color.y), channel(color.z)};
    }
    const double t = hotTemp > coldTemp ? (temperature - coldTemp) / (hotTemp - coldTemp) : 1.0;
    return heatColor(t).array().pow(2.2);
}

bool hasAlbedo(const HeatSurface& surface){
    return surface.width > 0 && surface.height > 0 &&
           surface.rgba.size() == static_cast<size_t>(surface.width) * surface.height * 4 &&
           surface.texcoords.size() == surface.positions.size();
}

// Meshes often map their top and bottom faces onto the same texels, so one texture could not cook the
// pan side differently from the top. The drawn albedo holds two copies side by side: the left page for
// triangles above GridFrame::midZ, the right page for those below it.
constexpr int kAlbedoPages = 2;

bool belowMid(const GridFrame& frame, float z){ return z < frame.midZ; }

std::shared_ptr<open3d::geometry::Image> makeAlbedo(const HeatSurface& surface){
    if (!hasAlbedo(surface)) return nullptr;
    auto image = std::make_shared<open3d::geometry::Image>();
    const int width = surface.width * kAlbedoPages;
    image->Prepare(width, surface.height, 3, 1);
    for (int y = 0; y < surface.height; ++y){
        for (int x = 0; x < width; ++x){
            const size_t src = (static_cast<size_t>(y) * surface.width + x % surface.width) * 4;
            const size_t dst = (static_cast<size_t>(y) * width + x) * 3;
            image->data_[dst] = surface.rgba[src];
            image->data_[dst + 1] = surface.rgba[src + 1];
            image->data_[dst + 2] = surface.rgba[src + 2];
        }
    }
    return image;
}

// Peak temperature at a model-space point, trilinear over the solid cells around it. Cells that are not
// meat get no weight, so a point on the surface takes the outermost meat layer's value.
float samplePeak(const HeatGrid& grid, const GridFrame& frame, glm::vec3 position){
    const glm::vec3 coord = (position - frame.origin) / frame.voxelSize - 0.5f;
    const glm::ivec3 base(glm::floor(coord));
    const glm::vec3 t = coord - glm::vec3(base);
    const auto solidAt = [&](glm::ivec3 c){
        return c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x < static_cast<int>(grid.dims.x) && c.y < static_cast<int>(grid.dims.y) &&
               c.z < static_cast<int>(grid.dims.z) &&
               grid.kind(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y), static_cast<uint32_t>(c.z)) == CELL_SOLID;
    };
    const auto peakAt = [&](glm::ivec3 c){
        return grid.peakCelsius[grid.index(static_cast<uint32_t>(c.x), static_cast<uint32_t>(c.y), static_cast<uint32_t>(c.z))];
    };
    float weight = 0.0f;
    float sum = 0.0f;
    for (int corner = 0; corner < 8; ++corner){
        const glm::ivec3 offset((corner & 1) ? 1 : 0, (corner & 2) ? 1 : 0, (corner & 4) ? 1 : 0);
        if (!solidAt(base + offset)) continue;
        const float w = (offset.x ? t.x : 1.0f - t.x) * (offset.y ? t.y : 1.0f - t.y) * (offset.z ? t.z : 1.0f - t.z);
        weight += w;
        sum += w * peakAt(base + offset);
    }
    if (weight > 1.0e-6f) return sum / weight;
    // Mesh and voxels disagree by up to a cell at thin spots; take the nearest meat within one cell.
    const glm::ivec3 nearest(glm::round(coord));
    float best = std::numeric_limits<float>::max();
    float peak = kNoPeakCelsius;
    for (int dz = -1; dz <= 1; ++dz){
        for (int dy = -1; dy <= 1; ++dy){
            for (int dx = -1; dx <= 1; ++dx){
                const glm::ivec3 c = nearest + glm::ivec3(dx, dy, dz);
                if (!solidAt(c)) continue;
                const float distance = glm::length(glm::vec3(c) - coord);
                if (distance < best){
                    best = distance;
                    peak = peakAt(c);
                }
            }
        }
    }
    return peak;
}

// Photo-resolution cook. Every triangle is rasterized into its albedo page; each covered texel finds its
// model-space point, samples the peak temperature there, and is cooked from its own Lab with the same
// doneness table as the voxels. The texture keeps its own detail, and the cook varies smoothly instead
// of in voxel-sized blocks.
std::shared_ptr<open3d::geometry::Image> bakeCookedAlbedo(const HeatGrid& grid, const HeatSurface& surface){
    auto image = makeAlbedo(surface);
    if (!image || grid.peakCelsius.size() != grid.cellCount()) return image;
    const GridFrame frame = gridFrame(grid, surface);
    const int pageWidth = surface.width;
    const int width = pageWidth * kAlbedoPages;
    const int height = surface.height;
    std::vector<float> peak(static_cast<size_t>(width) * height, std::numeric_limits<float>::quiet_NaN());

    for (size_t t = 0; t + 2 < surface.indices.size(); t += 3){
        const uint32_t ids[3] = {surface.indices[t], surface.indices[t + 1], surface.indices[t + 2]};
        if (ids[0] >= surface.positions.size() || ids[1] >= surface.positions.size() || ids[2] >= surface.positions.size()) continue;
        const glm::vec3 p[3] = {surface.positions[ids[0]], surface.positions[ids[1]], surface.positions[ids[2]]};
        const int page = belowMid(frame, (p[0].z + p[1].z + p[2].z) / 3.0f) ? 1 : 0;
        // Display UVs are V up; image rows run top first.
        glm::vec2 q[3];
        for (int c = 0; c < 3; ++c){
            const glm::vec2 uv = surface.texcoords[ids[c]];
            q[c] = {(uv.x + static_cast<float>(page)) * static_cast<float>(pageWidth), (1.0f - uv.y) * static_cast<float>(height)};
        }
        const float area = (q[1].x - q[0].x) * (q[2].y - q[0].y) - (q[2].x - q[0].x) * (q[1].y - q[0].y);
        if (!(std::abs(area) > 1.0e-8f)) continue;
        const int x0 = std::max(static_cast<int>(std::floor(std::min({q[0].x, q[1].x, q[2].x}))), page * pageWidth);
        const int x1 = std::min(static_cast<int>(std::ceil(std::max({q[0].x, q[1].x, q[2].x}))), (page + 1) * pageWidth - 1);
        const int y0 = std::max(static_cast<int>(std::floor(std::min({q[0].y, q[1].y, q[2].y}))), 0);
        const int y1 = std::min(static_cast<int>(std::ceil(std::max({q[0].y, q[1].y, q[2].y}))), height - 1);
        for (int y = y0; y <= y1; ++y){
            for (int x = x0; x <= x1; ++x){
                const glm::vec2 c(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
                const float b1 = ((c.x - q[0].x) * (q[2].y - q[0].y) - (q[2].x - q[0].x) * (c.y - q[0].y)) / area;
                const float b2 = ((q[1].x - q[0].x) * (c.y - q[0].y) - (c.x - q[0].x) * (q[1].y - q[0].y)) / area;
                const float b0 = 1.0f - b1 - b2;
                constexpr float kEdge = -1.0e-4f;
                if (b0 < kEdge || b1 < kEdge || b2 < kEdge) continue;
                peak[static_cast<size_t>(y) * width + x] = samplePeak(grid, frame, b0 * p[0] + b1 * p[1] + b2 * p[2]);
            }
        }
    }

    // Grow each island by a few texels so bilinear filtering at its edges does not pull in raw texels.
    constexpr int kBleed = 3;
    for (int pass = 0; pass < kBleed; ++pass){
        std::vector<float> grown = peak;
        for (int y = 0; y < height; ++y){
            for (int x = 0; x < width; ++x){
                const size_t texel = static_cast<size_t>(y) * width + x;
                if (!std::isnan(peak[texel])) continue;
                const int neighbor[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (const auto& step : neighbor){
                    const int nx = x + step[0];
                    const int ny = y + step[1];
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height || nx / pageWidth != x / pageWidth) continue;
                    const float value = peak[static_cast<size_t>(ny) * width + nx];
                    if (!std::isnan(value)){
                        grown[texel] = value;
                        break;
                    }
                }
            }
        }
        peak.swap(grown);
    }

    const ColorReference reference{grid.colorReferenceY, grid.colorReferenceA};
    for (size_t texel = 0; texel < peak.size(); ++texel){
        if (std::isnan(peak[texel])) continue;
        const size_t byte = texel * 3;
        const glm::vec3 encoded{image->data_[byte] / 255.0f, image->data_[byte + 1] / 255.0f, image->data_[byte + 2] / 255.0f};
        const glm::vec3 lab0 = linearSrgbToLab(srgbToLinear(encoded));
        const glm::vec3 out = linearToSrgb(labToLinearSrgb(cookedLab(lab0, peak[texel], reference)));
        image->data_[byte] = static_cast<uint8_t>(std::lround(out.x * 255.0f));
        image->data_[byte + 1] = static_cast<uint8_t>(std::lround(out.y * 255.0f));
        image->data_[byte + 2] = static_cast<uint8_t>(std::lround(out.z * 255.0f));
    }
    return image;
}

// Textured mesh, the same surface steak_color_demo draws. Cooked color is a full-resolution
// texture; vertex colors are only the fallback when the mesh has no albedo.
open3d::geometry::TriangleMesh buildSurfaceMesh(const HeatGrid& grid, const HeatSurface& surface, int sliceAxis, int slice,
                                                float coldTemp, float hotTemp, bool showCookedColor){
    open3d::geometry::TriangleMesh mesh;
    if (!hasSurface(surface) || !(grid.spacing > 0.0f)) return mesh;
    const GridFrame frame = gridFrame(grid, surface);
    if (!(frame.voxelSize > 0.0f)) return mesh;
    const double mmPerModel = grid.spacing * MM_PER_M / frame.voxelSize;
    const bool textured = hasAlbedo(surface);
    const auto cellOf = [&](glm::vec3 position){ return (position - frame.origin) / frame.voxelSize; };
    const auto inSlice = [&](glm::vec3 position){
        if (slice < 0) return true;
        const float c = cellOf(position)[sliceAxis];
        return c >= static_cast<float>(slice) && c < static_cast<float>(slice) + 1.0f;
    };

    std::vector<int> remap(surface.positions.size(), -1);
    for (size_t t = 0; t + 2 < surface.indices.size(); t += 3){
        const uint32_t ids[3] = {surface.indices[t], surface.indices[t + 1], surface.indices[t + 2]};
        if (ids[0] >= surface.positions.size() || ids[1] >= surface.positions.size() || ids[2] >= surface.positions.size()) continue;
        if (!inSlice(surface.positions[ids[0]]) && !inSlice(surface.positions[ids[1]]) && !inSlice(surface.positions[ids[2]])) continue;
        Eigen::Vector3i triangle;
        const float centroidZ =
            (surface.positions[ids[0]].z + surface.positions[ids[1]].z + surface.positions[ids[2]].z) / 3.0f;
        const float page = belowMid(frame, centroidZ) ? 1.0f : 0.0f;
        for (int corner = 0; corner < 3; ++corner){
            int& slot = remap[ids[corner]];
            if (slot < 0){
                const glm::vec3 position = surface.positions[ids[corner]];
                const glm::vec3 placed = (position - frame.origin) * static_cast<float>(mmPerModel);
                mesh.vertices_.emplace_back(placed.x, placed.y, placed.z);
                if (!textured) mesh.vertex_colors_.push_back(sampleVoxel(grid, frame, position, coldTemp, hotTemp, showCookedColor));
                slot = static_cast<int>(mesh.vertices_.size()) - 1;
            }
            triangle[corner] = slot;
            if (textured){
                const glm::vec2 uv = surface.texcoords[ids[corner]];
                mesh.triangle_uvs_.emplace_back((uv.x + page) / kAlbedoPages, uv.y);
            }
        }
        mesh.triangles_.push_back(triangle);
    }
    if (!mesh.triangles_.empty()) mesh.ComputeVertexNormals();
    return mesh;
}

class HeatViewWindow : public gui::Window{
public:
    HeatViewWindow(const std::string& title, HeatGrid gridIn, HeatParams paramsIn, uint32_t stepsIn, HeatSimulator simulateIn,
                   bool gridFollowsMeshIn, HeatSurface surfaceIn, gui::FontId headingFontIn)
        : gui::Window(title, 1400, 850),
          grid(std::move(gridIn)),
          edited(paramsIn),
          simulate(std::move(simulateIn)),
          surface(std::move(surfaceIn)),
          editedSteps(stepsIn),
          gridFollowsMesh(gridFollowsMeshIn),
          headingFont(headingFontIn){
        coldTemp = COLOR_MIN_F;
        hotTemp = COLOR_MAX_F;
        recount();
        // Stock text boxes are nearly black. A lighter fill keeps the number readable.
        const_cast<gui::Theme&>(GetTheme()).text_edit_background_color = gui::Color(0.42f, 0.44f, 0.48f);

        scene = std::make_shared<gui::SceneWidget>();
        scene->SetScene(std::make_shared<rendering::Open3DScene>(GetRenderer()));
        scene->GetScene()->SetBackground({0.08f, 0.08f, 0.10f, 1.0f});
        scene->GetScene()->GetView()->SetColorGrading(rendering::ColorGradingParams(
            rendering::ColorGradingParams::Quality::kHigh, rendering::ColorGradingParams::ToneMapping::kLinear));
        material.shader = "defaultUnlit";
        textureMaterial.shader = "defaultUnlit";
        albedoImage = makeAlbedo(surface);
        textureMaterial.albedo_img = albedoImage;
        textureMaterial.sRGB_color = true;
        rebuildView();

        const auto& bounds = scene->GetScene()->GetBoundingBox();
        scene->SetupCamera(60.0f, bounds, bounds.GetCenter().cast<float>());
        AddChild(scene);
        AddChild(buildPanel());

        SetOnKeyEvent([this](const gui::KeyEvent& e){ return onKey(e); });
        SetOnClose([this](){ return !resetting; });
        SetOnTickEvent([this](){ return onTick(); });
    }

    void Layout(const gui::LayoutContext& context) override{
        const gui::Rect r = GetContentRect();
        const int panelWidth = std::min(r.width / 2, 34 * context.theme.font_size);
        panel->SetFrame(gui::Rect(r.GetRight() - panelWidth, r.y, panelWidth, r.height));
        scene->SetFrame(gui::Rect(r.x, r.y, r.width - panelWidth, r.height));
        gui::Window::Layout(context);
    }

private:
    struct SettingField{
        std::function<std::string()> current;
        std::function<bool(const std::string&, std::string&)> commit;
        std::shared_ptr<gui::TextEdit> edit;
        std::shared_ptr<gui::Label> value;
    };

    void recount(){
        solidCells = 0;
        panContactCells = 0;
        minTemp = std::numeric_limits<float>::max();
        maxTemp = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < grid.cellCount(); ++i){
            if (grid.cells[i] != CELL_SOLID) continue;
            ++solidCells;
            minTemp = std::min(minTemp, grid.temperature[i]);
            maxTemp = std::max(maxTemp, grid.temperature[i]);
            const size_t slab = size_t(grid.dims.x) * grid.dims.y;
            if (i >= slab && grid.cells[i - slab] == CELL_PAN) ++panContactCells;
        }
    }

    void ensureCookedAlbedo(){
        if (cookedReady) return;
        cookedAlbedo = bakeCookedAlbedo(grid, surface);
        cookedReady = true;
    }

    void uploadView(){
        auto& scene3d = *scene->GetScene();
        if (scene3d.HasGeometry("blocks")) scene3d.RemoveGeometry("blocks");
        if (scene3d.HasGeometry("surface")) scene3d.RemoveGeometry("surface");
        if (showVolume || !hasSurface(surface)){
            if (!blocks.triangles_.empty()) scene3d.AddGeometry("blocks", &blocks, material, false);
        } else if (!surfaceDraw.triangles_.empty()){
            if (showCookedColor && hasAlbedo(surface)) ensureCookedAlbedo();
            const bool photo = textureMaterial.albedo_img != nullptr;
            if (photo) textureMaterial.albedo_img = (showCookedColor && cookedAlbedo) ? cookedAlbedo : albedoImage;
            scene3d.AddGeometry("surface", &surfaceDraw, photo ? textureMaterial : material, false);
        }
        PostRedraw();
    }

    void rebuildView(){
        if (resetting) return;
        if (showVolume || !hasSurface(surface)){
            blocks = buildHeatMesh(grid, sliceAxis, slice, showPan, coldTemp, hotTemp, showCookedColor, showInitialColor, {});
        } else{
            surfaceDraw = buildSurfaceMesh(grid, surface, sliceAxis, slice, coldTemp, hotTemp, showCookedColor);
        }
        uploadView();
    }

    void syncSliceControls(){
        const bool on = slice >= 0;
        singleSlice->SetChecked(on);
        sliceSlider->SetEnabled(on);
        axisCombo->SetEnabled(on);
        axisLabel->SetTextColor(on ? gui::Color(0.9f, 0.9f, 0.9f) : gui::Color(0.45f, 0.45f, 0.45f));
        if (on) sliceSlider->SetValue(slice);
    }

    void setSlice(int newSlice){
        slice = std::clamp(newSlice, -1, static_cast<int>(grid.dims[sliceAxis]) - 1);
        syncSliceControls();
        rebuildView();
    }

    void setSliceAxis(int axis){
        sliceAxis = axis;
        axisCombo->SetSelectedIndex(axis);
        sliceSlider->SetLimits(0, grid.dims[axis] - 1);
        setSlice(slice < 0 ? -1 : static_cast<int>(grid.dims[axis]) / 2);
    }

    void updateLegend(){
        for (int row = 0; row < LEGEND_ROWS; ++row){
            const double t = 1.0 - static_cast<double>(row) / (LEGEND_ROWS - 1);
            const std::string prefix = row == 0 ? ">= " : row == LEGEND_ROWS - 1 ? "<= " : "";
            legendLabels[row]->SetText((prefix + format("%.0f", coldTemp + t * (hotTemp - coldTemp))).c_str());
        }
    }

    void setColorMax(float value){
        hotTemp = value;
        updateLegend();
        rebuildView();
    }

    void setShowPan(bool show){
        showPan = show;
        panCheckbox->SetChecked(show);
        rebuildView();
    }

    void setShowCookedColor(bool show){
        if (grid.color.size() != grid.cellCount() || showCookedColor == show) return;
        showCookedColor = show;
        if (show && showInitialColor){
            showInitialColor = false;
            if (initialCheckbox) initialCheckbox->SetChecked(false);
        }
        if (colorCheckbox) colorCheckbox->SetChecked(show);
        rebuildView();
    }

    void setShowInitialColor(bool show){
        if (grid.initialColor.size() != grid.cellCount() || showInitialColor == show) return;
        showInitialColor = show;
        if (show){
            showCookedColor = false;
            showVolume = true;
            if (colorCheckbox) colorCheckbox->SetChecked(false);
            if (volumeCheckbox) volumeCheckbox->SetChecked(true);
        }
        if (initialCheckbox) initialCheckbox->SetChecked(show);
        rebuildView();
    }

    void setShowVolume(bool show){
        if (!hasSurface(surface) || showVolume == show) return;
        showVolume = show;
        if (volumeCheckbox) volumeCheckbox->SetChecked(show);
        rebuildView();
    }

    void refreshFields(){
        for (SettingField& field : fields){
            const std::string text = field.current();
            if (field.edit) field.edit->SetText(text.c_str());
            if (field.value) field.value->SetText(text.c_str());
        }
    }

    void onFieldChanged(size_t index, const char* text){
        std::string error;
        if (!fields[index].commit(text, error)) ShowMessageBox("Invalid value", error.c_str());
        refreshFields();
    }

    // Writes `edited` only after the whole value is valid, so a rejected edit leaves the saved model alone.
    bool commitSteps(long count, std::string& error){
        if (count < 1 || count > MAX_STEPS){
            error = "Steps must be a whole number from 1 to " + std::to_string(MAX_STEPS) + ".";
            return false;
        }
        editedSteps = static_cast<uint32_t>(count);
        edited.duration = static_cast<float>(editedSteps) * edited.timeStep;
        return true;
    }

    bool commitDurationOrStep(double duration, double timeStep, std::string& error){
        if (!(duration > 0.0) || !(timeStep > 0.0)){
            error = "Duration and time step must be positive.";
            return false;
        }
        const double count = std::round(duration / timeStep);
        if (count < 1.0 || count > MAX_STEPS){
            error = "That duration and time step need between 1 and " + std::to_string(MAX_STEPS) + " steps.";
            return false;
        }
        edited.duration = static_cast<float>(duration);
        edited.timeStep = static_cast<float>(timeStep);
        editedSteps = static_cast<uint32_t>(count);
        return true;
    }

    std::string resetError() const{
        if (editedSteps < 1 || editedSteps > MAX_STEPS) return "Steps must be from 1 to " + std::to_string(MAX_STEPS) + ".";
        if (!(edited.spacing > 0.0f) || !(edited.timeStep > 0.0f) || !(edited.diffusivity > 0.0f) || !(edited.conductivity > 0.0f) ||
            edited.panTransfer < 0.0f || edited.airTransfer < 0.0f){
            return "Spacing, time step, diffusivity and conductivity must be positive, and heat transfer cannot be negative.";
        }
        if (!gridFollowsMesh && (edited.dims.x == 0 || edited.dims.y == 0 || edited.dims.z == 0)) return "The grid is empty.";
        if (edited.fourier() > edited.maxStableFourier()){
            const float maxStep = edited.maxStableFourier() * edited.spacing * edited.spacing / edited.diffusivity;
            return "Unstable: alpha*dt/dx^2 = " + format("%.4g", edited.fourier()) + ", above " +
                   format("%.4g", edited.maxStableFourier()) + ". Use a time step of " + format("%.4g", maxStep) + " s or less.";
        }
        return {};
    }

    void addField(gui::Widget& parent, std::string caption, bool editable, std::function<std::string()> current,
                  std::function<bool(const std::string&, std::string&)> commit){
        const size_t index = fields.size();
        SettingField field;
        field.current = std::move(current);
        field.commit = std::move(commit);

        auto block = std::make_shared<gui::Vert>(2);
        auto label = std::make_shared<gui::Label>(caption.c_str());
        label->SetTextColor(gui::Color(0.82f, 0.82f, 0.82f));
        block->AddChild(label);
        if (editable){
            field.edit = std::make_shared<gui::TextEdit>();
            field.edit->SetText(field.current().c_str());
            field.edit->SetOnValueChanged([this, index](const char* text){ onFieldChanged(index, text); });
            block->AddChild(field.edit);
        } else{
            field.value = std::make_shared<gui::Label>(field.current().c_str());
            block->AddChild(field.value);
        }
        parent.AddChild(block);
        fields.push_back(std::move(field));
    }

    std::shared_ptr<steakui::Disclosure> makeSection(const char* title){
        auto section = std::make_shared<steakui::Disclosure>(title, headingFont, GetTheme().font_size);
        section->SetOnToggled([this](){
            Layout(GetLayoutContext());
            PostRedraw();
        });
        return section;
    }

    std::shared_ptr<gui::Widget> buildPanel(){
        const int em = GetTheme().font_size;
        panel = std::make_shared<gui::Vert>(em / 2, gui::Margins(em));
        fields.reserve(24);

        panel->AddChild(std::make_shared<gui::Label>("Temperature (F)"));
        for (int row = 0; row < LEGEND_ROWS; ++row){
            const double t = 1.0 - static_cast<double>(row) / (LEGEND_ROWS - 1);
            auto line = std::make_shared<gui::Horiz>(em / 2);
            line->AddChild(std::make_shared<steakui::Chrome>(makeSwatch(heatColor(t), 2 * em, em)));
            legendLabels[row] = std::make_shared<gui::Label>("");
            line->AddChild(legendLabels[row]);
            panel->AddChild(line);
        }
        updateLegend();

        auto rangeRow = std::make_shared<gui::Horiz>(em);
        rangeRow->AddChild(std::make_shared<gui::Label>("Color max"));
        rangeSlider = std::make_shared<steakui::TrackSlider>();
        rangeSlider->SetLimits(coldTemp + 10.0, std::max(COLOR_MAX_F, edited.panTemp));
        rangeSlider->SetValue(hotTemp);
        rangeSlider->SetOnValueChanged([this](double value){ setColorMax(static_cast<float>(value)); });
        rangeRow->AddChild(rangeSlider);
        panel->AddChild(rangeRow);
        panel->AddFixed(em / 2);

        auto resetButton = std::make_shared<gui::Button>("Reset simulation");
        resetButton->SetOnClicked([this](){ askReset(); });
        panel->AddChild(resetButton);
        panel->AddFixed(em / 2);

        auto settings = std::make_shared<gui::Label>("Settings");
        settings->SetFontId(headingFont);
        settings->SetTextColor(gui::Color(1.0f, 1.0f, 1.0f));
        panel->AddChild(settings);
        auto hint = std::make_shared<gui::Label>("Type a value, then reset to run the simulation.");
        hint->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        panel->AddChild(hint);

        auto scroll = std::make_shared<gui::ScrollableVert>(em / 2, gui::Margins(0, em / 3, em + em / 2, 0));

        auto view = makeSection("View");
        auto& viewBody = view->content();
        panCheckbox = std::make_shared<gui::Checkbox>("Show pan");
        panCheckbox->SetChecked(showPan);
        panCheckbox->SetOnChecked([this](bool checked){ setShowPan(checked); });
        viewBody.AddChild(panCheckbox);
        if (grid.color.size() == grid.cellCount()){
            colorCheckbox = std::make_shared<gui::Checkbox>("Cooked color");
            colorCheckbox->SetChecked(showCookedColor);
            colorCheckbox->SetOnChecked([this](bool checked){ setShowCookedColor(checked); });
            viewBody.AddChild(colorCheckbox);
            initialCheckbox = std::make_shared<gui::Checkbox>("Initial texture");
            initialCheckbox->SetChecked(showInitialColor);
            initialCheckbox->SetOnChecked([this](bool checked){ setShowInitialColor(checked); });
            viewBody.AddChild(initialCheckbox);
        }
        if (hasSurface(surface)){
            volumeCheckbox = std::make_shared<gui::Checkbox>("Voxel volume");
            volumeCheckbox->SetChecked(showVolume);
            volumeCheckbox->SetOnChecked([this](bool checked){ setShowVolume(checked); });
            viewBody.AddChild(volumeCheckbox);
        }
        singleSlice = std::make_shared<gui::Checkbox>("Single slice");
        singleSlice->SetOnChecked([this](bool checked){ setSlice(checked ? static_cast<int>(grid.dims[sliceAxis]) / 2 : -1); });
        viewBody.AddChild(singleSlice);
        sliceSlider = std::make_shared<steakui::TrackSlider>();
        sliceSlider->SetLimits(0, grid.dims[sliceAxis] - 1);
        sliceSlider->SetEnabled(false);
        sliceSlider->SetOnValueChanged([this](double value){ setSlice(static_cast<int>(value)); });
        viewBody.AddChild(sliceSlider);
        axisLabel = std::make_shared<gui::Label>("Slice Axis");
        axisLabel->SetTextColor(gui::Color(0.45f, 0.45f, 0.45f));
        axisCombo = std::make_shared<gui::Combobox>(std::vector<const char*>(AXIS_NAMES.begin(), AXIS_NAMES.end()));
        axisCombo->SetSelectedIndex(sliceAxis);
        axisCombo->SetEnabled(false);
        axisCombo->SetOnValueChanged([this](const char*, int index){ setSliceAxis(index); });
        viewBody.AddChild(std::make_shared<steakui::SliceAxisRow>(axisLabel, axisCombo));
        scroll->AddChild(view);

        auto time = makeSection("Time");
        auto& timeBody = time->content();
        addField(timeBody, "Duration (s)", true, [this](){ return format("%.6g", edited.duration); }, [this](const std::string& text, std::string& error){
            double value = 0.0;
            return parseNumber(text.c_str(), value, error) && commitDurationOrStep(value, edited.timeStep, error);
        });
        addField(timeBody, "Time step (s)", true, [this](){ return format("%.6g", edited.timeStep); }, [this](const std::string& text, std::string& error){
            double value = 0.0;
            return parseNumber(text.c_str(), value, error) && commitDurationOrStep(edited.duration, value, error);
        });
        addField(timeBody, "Steps", true, [this](){ return std::to_string(editedSteps); }, [this](const std::string& text, std::string& error){
            long count = 0;
            return parseCount(text.c_str(), count, 1, MAX_STEPS, error, "Steps") && commitSteps(count, error);
        });
        scroll->AddChild(time);

        auto gridSection = makeSection("Grid");
        auto& gridBody = gridSection->content();
        const char* dimNames[] = {"Cells X", "Cells Y", "Cells Z"};
        for (int axis = 0; axis < 3; ++axis){
            addField(gridBody, dimNames[axis], !gridFollowsMesh,
                     [this, axis](){
                         const glm::uvec3 dims = gridFollowsMesh ? grid.dims : edited.dims;
                         return std::to_string(dims[axis]);
                     },
                     [this, axis](const std::string& text, std::string& error){
                         long count = 0;
                         if (!parseCount(text.c_str(), count, 1, MAX_DIM, error, "Cells")) return false;
                         edited.dims[axis] = static_cast<uint32_t>(count);
                         return true;
                     });
        }
        addField(gridBody, "Cell size (mm)", true, [this](){ return format("%.6g", edited.spacing * MM_PER_M); },
                 [this](const std::string& text, std::string& error){
                     double mm = 0.0;
                     if (!parseNumber(text.c_str(), mm, error)) return false;
                     if (!(mm > 0.0)){
                         error = "Cell size must be positive.";
                         return false;
                     }
                     edited.spacing = static_cast<float>(mm / MM_PER_M);
                     return true;
                 });
        addField(gridBody, "Solid cells", false, [this](){ return std::to_string(solidCells); }, {});
        addField(gridBody, "On the pan", false, [this](){ return std::to_string(panContactCells); }, {});
        scroll->AddChild(gridSection);

        auto pan = makeSection("Pan");
        auto& panBody = pan->content();
        addField(panBody, "Temperature (F)", true, [this](){ return format("%.6g", edited.panTemp); }, [this](const std::string& text, std::string& error){
            double value = 0.0;
            if (!parseNumber(text.c_str(), value, error)) return false;
            edited.panTemp = static_cast<float>(value);
            return true;
        });
        addField(panBody, "Heat transfer (W/m^2K)", true, [this](){ return format("%.6g", edited.panTransfer); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     if (value < 0.0){
                         error = "Heat transfer cannot be negative.";
                         return false;
                     }
                     edited.panTransfer = static_cast<float>(value);
                     return true;
                 });
        scroll->AddChild(pan);

        auto air = makeSection("Air");
        auto& airBody = air->content();
        addField(airBody, "Temperature (F)", true, [this](){ return format("%.6g", edited.airTemp); }, [this](const std::string& text, std::string& error){
            double value = 0.0;
            if (!parseNumber(text.c_str(), value, error)) return false;
            edited.airTemp = static_cast<float>(value);
            return true;
        });
        addField(airBody, "Heat transfer (W/m^2K)", true, [this](){ return format("%.6g", edited.airTransfer); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     if (value < 0.0){
                         error = "Heat transfer cannot be negative.";
                         return false;
                     }
                     edited.airTransfer = static_cast<float>(value);
                     return true;
                 });
        scroll->AddChild(air);

        auto meat = makeSection("Meat");
        auto& meatBody = meat->content();
        addField(meatBody, "Start temperature (F)", true, [this](){ return format("%.6g", edited.initialTemp); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     edited.initialTemp = static_cast<float>(value);
                     return true;
                 });
        addField(meatBody, "Temperature range (F)", false, [this](){
            if (solidCells == 0) return std::string("n/a");
            return format("%.4g", minTemp) + " to " + format("%.4g", maxTemp);
        }, {});
        addField(meatBody, "Diffusivity (m^2/s)", true, [this](){ return format("%.6g", edited.diffusivity); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     if (!(value > 0.0)){
                         error = "Diffusivity must be positive.";
                         return false;
                     }
                     edited.diffusivity = static_cast<float>(value);
                     return true;
                 });
        addField(meatBody, "Conductivity (W/mK)", true, [this](){ return format("%.6g", edited.conductivity); },
                 [this](const std::string& text, std::string& error){
                     double value = 0.0;
                     if (!parseNumber(text.c_str(), value, error)) return false;
                     if (!(value > 0.0)){
                         error = "Conductivity must be positive.";
                         return false;
                     }
                     edited.conductivity = static_cast<float>(value);
                     return true;
                 });
        addField(meatBody, "Fourier r", false, [this](){ return format("%.4g", edited.fourier()); }, {});
        scroll->AddChild(meat);

        std::string helpText = "Drag to rotate, scroll to zoom\n Press 'X', 'Y', 'Z' to take slices along axes and move them with the arrow keys\n Press 'A' to toggle back to the full domain. \n Press 'P' to toggle pan.";
        if (grid.color.size() == grid.cellCount()){
            helpText += "\n Press 'C' to toggle cooked color.";
            helpText += "\n Press 'I' for the voxel volume of the texture before cooking.";
        }
        if (hasSurface(surface)) helpText += "\n Press 'V' to toggle the voxel volume and the mesh.";
        auto help = std::make_shared<gui::Label>(helpText.c_str());
        help->SetTextColor(gui::Color(0.7f, 0.7f, 0.7f));
        scroll->AddChild(help);
        panel->AddChild(scroll);
        return panel;
    }

    void askReset(){
        if (resetting) return;
        const std::string error = resetError();
        if (!error.empty()){
            ShowMessageBox("Cannot reset", error.c_str());
            return;
        }
        const int width = std::max(320, 28 * GetTheme().font_size);
        auto dialog = std::make_shared<steakui::ConfirmDialog>(
            "Reset simulation", "Reset the simulation with the saved settings?", width, [this](bool accepted){
                gui::Application::GetInstance().PostToMainThread(this, [this, accepted](){
                    steakui::chromeHidden() = false;
                    CloseDialog();
                    if (accepted) startReset();
                });
            });
        steakui::chromeHidden() = true;
        ShowDialog(dialog);
    }

    void startReset(){
        if (resetting) return;
        resetting = true;
        const int width = std::max(320, 28 * GetTheme().font_size);
        progressDialog = std::make_shared<steakui::ResetProgressDialog>("Resetting the simulation.", width);
        progressDialog->SetFraction(0.0f);
        steakui::chromeHidden() = true;
        ShowDialog(progressDialog);

        const HeatParams ran = edited;
        const uint32_t ranSteps = editedSteps;
        const int axis = sliceAxis;
        const int sliceNow = slice;
        const bool pan = showPan;
        const bool cooked = showCookedColor;
        const bool initial = showInitialColor;
        const float cold = coldTemp;
        const float hot = hotTemp;
        auto dialog = progressDialog;
        auto simulator = simulate;
        gui::Application::GetInstance().RunInThread([this, dialog, ran, ranSteps, axis, sliceNow, pan, cooked, initial, cold, hot, simulator](){
            int lastPercent = -1;
            auto report = [this, dialog, &lastPercent](float fraction){
                const int percent = std::clamp(static_cast<int>(std::lround(fraction * 100.0f)), 0, 100);
                if (percent == lastPercent) return;
                lastPercent = percent;
                gui::Application::GetInstance().PostToMainThread(this, [dialog, percent](){ dialog->SetFraction(percent / 100.0f); });
            };
            try{
                report(0.01f);
                HeatGrid next = simulator(ran, ranSteps, [&](float fraction){ report(0.02f + 0.78f * fraction); });
                auto mesh = std::make_shared<open3d::geometry::TriangleMesh>(
                    buildHeatMesh(next, axis, sliceNow, pan, cold, hot, cooked, initial, [&](float fraction){ report(0.80f + 0.19f * fraction); }));
                auto gridPtr = std::make_shared<HeatGrid>(std::move(next));
                auto applied = std::make_shared<HeatParams>(ran);
                gui::Application::GetInstance().PostToMainThread(
                    this, [this, gridPtr, applied, ranSteps, axis, sliceNow, mesh](){ finishReset(*gridPtr, *applied, ranSteps, axis, sliceNow, *mesh); });
            } catch (const std::exception& e){
                const std::string message = e.what();
                gui::Application::GetInstance().PostToMainThread(this, [this, message](){
                    steakui::chromeHidden() = false;
                    CloseDialog();
                    progressDialog.reset();
                    resetting = false;
                    ShowMessageBox("Reset failed", message.c_str());
                });
            }
        });
    }

    void finishReset(HeatGrid next, HeatParams ran, uint32_t ranSteps, int axis, int sliceNow, open3d::geometry::TriangleMesh mesh){
        const bool shapeChanged = next.dims != grid.dims || next.spacing != grid.spacing;
        grid = std::move(next);
        edited = ran;
        edited.dims = grid.dims;
        editedSteps = ranSteps;
        recount();

        sliceAxis = axis;
        slice = sliceNow < 0 ? -1 : std::min(sliceNow, std::max(0, static_cast<int>(grid.dims[sliceAxis]) - 1));
        blocks = std::move(mesh);
        cookedReady = false;
        cookedAlbedo.reset();
        if (!showVolume && hasSurface(surface)){
            surfaceDraw = buildSurfaceMesh(grid, surface, sliceAxis, slice, coldTemp, hotTemp, showCookedColor);
        }
        uploadView();
        if (shapeChanged && !blocks.triangles_.empty()){
            const auto& bounds = scene->GetScene()->GetBoundingBox();
            scene->SetupCamera(60.0f, bounds, bounds.GetCenter().cast<float>());
        }

        axisCombo->SetSelectedIndex(sliceAxis);
        sliceSlider->SetLimits(0, std::max(1u, grid.dims[sliceAxis]) - 1u);
        syncSliceControls();
        const double upper = std::max({static_cast<double>(COLOR_MAX_F), static_cast<double>(edited.panTemp), static_cast<double>(hotTemp),
                                       coldTemp + 10.0});
        rangeSlider->SetLimits(coldTemp + 10.0, upper);
        refreshFields();
        progressDialog->SetFraction(1.0f);
        progressDoneAt = gui::Application::GetInstance().Now() + 0.25;
        progressArmed = true;
    }

    bool onTick(){
        if (!progressArmed) return false;
        if (gui::Application::GetInstance().Now() < progressDoneAt) return true;
        steakui::chromeHidden() = false;
        CloseDialog();
        progressDialog.reset();
        progressArmed = false;
        resetting = false;
        return true;
    }

    bool onKey(const gui::KeyEvent& e){
        if (e.type != gui::KeyEvent::DOWN) return false;
        if (resetting) return true;
        const int middle = static_cast<int>(grid.dims[sliceAxis]) / 2;
        switch (e.key){
            case gui::KEY_X: setSliceAxis(0); return true;
            case gui::KEY_Y: setSliceAxis(1); return true;
            case gui::KEY_Z: setSliceAxis(2); return true;
            case gui::KEY_UP: setSlice(slice < 0 ? middle : slice + 1); return true;
            case gui::KEY_DOWN: setSlice(slice < 0 ? middle : std::max(slice - 1, 0)); return true;
            case gui::KEY_A: setSlice(-1); return true;
            case gui::KEY_P: setShowPan(!showPan); return true;
            case gui::KEY_C: setShowCookedColor(!showCookedColor); return true;
            case gui::KEY_I: setShowInitialColor(!showInitialColor); return true;
            case gui::KEY_V: setShowVolume(!showVolume); return true;
            default: return false;
        }
    }

    HeatGrid grid;
    HeatParams edited;
    HeatSimulator simulate;
    HeatSurface surface;
    uint32_t editedSteps = 0;
    bool gridFollowsMesh = false;
    gui::FontId headingFont = 0;
    float coldTemp = 0.0f, hotTemp = 0.0f;
    float minTemp = 0.0f, maxTemp = 0.0f;
    size_t solidCells = 0, panContactCells = 0;
    int sliceAxis = 2;
    int slice = -1;
    bool showPan = true;
    bool showCookedColor = false;
    bool showInitialColor = false;
    bool showVolume = true;
    bool resetting = false;
    bool progressArmed = false;
    double progressDoneAt = 0.0;
    open3d::geometry::TriangleMesh blocks;
    open3d::geometry::TriangleMesh surfaceDraw;
    rendering::MaterialRecord material;
    rendering::MaterialRecord textureMaterial;
    std::shared_ptr<open3d::geometry::Image> albedoImage;
    std::shared_ptr<open3d::geometry::Image> cookedAlbedo;
    bool cookedReady = false;
    std::shared_ptr<gui::SceneWidget> scene;
    std::shared_ptr<gui::Vert> panel;
    std::array<std::shared_ptr<gui::Label>, LEGEND_ROWS> legendLabels;
    std::shared_ptr<gui::Checkbox> panCheckbox;
    std::shared_ptr<gui::Checkbox> colorCheckbox;
    std::shared_ptr<gui::Checkbox> initialCheckbox;
    std::shared_ptr<gui::Checkbox> volumeCheckbox;
    std::shared_ptr<gui::Label> axisLabel;
    std::shared_ptr<gui::Combobox> axisCombo;
    std::shared_ptr<gui::Checkbox> singleSlice;
    std::shared_ptr<steakui::TrackSlider> sliceSlider;
    std::shared_ptr<steakui::TrackSlider> rangeSlider;
    std::shared_ptr<steakui::ResetProgressDialog> progressDialog;
    std::vector<SettingField> fields;
};

gui::FontId headingFont(){
    try{
        return gui::Application::GetInstance().AddFont(gui::FontDescription(gui::FontDescription::SANS_SERIF, gui::FontStyle::BOLD, 20));
    } catch (const std::exception& e){
        std::cerr << "heat view: heading font unavailable (" << e.what() << ")\n";
        return gui::Application::DEFAULT_FONT_ID;
    }
}

}  // namespace

void printCenterColumn(const HeatGrid& grid, uint32_t rows){
    const uint32_t cx = grid.dims.x / 2, cy = grid.dims.y / 2;
    std::cout << "Temperature up the center column at t = " << grid.time << " s:\n";
    for (uint32_t z = 0; z < std::min(grid.dims.z, rows); ++z){
        const uint32_t kind = grid.kind(cx, cy, z);
        std::cout << "  z = " << format("%5.1f", z * grid.spacing * MM_PER_M) << " mm: " << format("%6.1f", grid.at(cx, cy, z)) << " F"
                  << (kind == CELL_PAN ? "  (pan)" : kind == CELL_AIR ? "  (air)" : "") << "\n";
    }
    std::cout << std::flush;
}

void showHeatGrid(const std::string& title, HeatGrid grid, const HeatParams& params, uint32_t steps, HeatSimulator simulate,
                  bool gridFollowsMesh, HeatSurface surface){
    auto& app = gui::Application::GetInstance();
    app.Initialize(OPEN3D_RESOURCE_DIR);
    const gui::FontId font = headingFont();
    app.AddWindow(std::make_shared<HeatViewWindow>(title, std::move(grid), params, steps, std::move(simulate), gridFollowsMesh,
                                                   std::move(surface), font));
    app.Run();
}
