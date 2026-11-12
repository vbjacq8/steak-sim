#include "demo_mesh.hpp"

#include "steak_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {

RgbaImage loadRgba(const std::string& path){
    int width = 0, height = 0, channels = 0;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
    if (!pixels) throw std::runtime_error("failed to load " + path);
    RgbaImage image{width, height, std::vector<uint8_t>(pixels, pixels + static_cast<size_t>(width) * height * 4)};
    stbi_image_free(pixels);
    return image;
}

// OBJ stores V pointing up. The albedo image is uploaded top row first, so flip OBJ V to match steak_sim.
std::vector<glm::vec2> albedoTexcoords(const Mesh& mesh){
    if (mesh.texcoords.size() != mesh.positions.size()) throw std::runtime_error("model has no texture coordinates");
    std::vector<glm::vec2> uvs = mesh.texcoords;
    if (mesh.format == MeshFormat::Obj){
        for (glm::vec2& uv : uvs) uv.y = 1.0f - uv.y;
    }
    return uvs;
}

RgbaImage decodeMemory(const uint8_t* data, int size, const std::string& name){
    int width = 0, height = 0, channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(data, size, &width, &height, &channels, 4);
    if (!pixels){
        const char* reason = stbi_failure_reason();
        throw std::runtime_error("failed to decode " + name + (reason ? std::string(": ") + reason : std::string()));
    }
    RgbaImage image{width, height, std::vector<uint8_t>(pixels, pixels + static_cast<size_t>(width) * height * 4)};
    stbi_image_free(pixels);
    return image;
}

// A steak lies on its broad side, so the pan goes against its thinnest extent.
uint32_t thinnestAxis(const std::vector<glm::vec3>& positions){
    glm::vec3 low(std::numeric_limits<float>::infinity());
    glm::vec3 high(-std::numeric_limits<float>::infinity());
    for (const glm::vec3& point : positions){
        low = glm::min(low, point);
        high = glm::max(high, point);
    }
    const glm::vec3 extent = high - low;
    return extent.x <= extent.y && extent.x <= extent.z ? 0u : extent.y <= extent.z ? 1u : 2u;
}

}  // namespace

float longestAxis(const std::vector<glm::vec3>& positions){
    glm::vec3 low(std::numeric_limits<float>::infinity());
    glm::vec3 high(-std::numeric_limits<float>::infinity());
    for (const glm::vec3& point : positions){
        low = glm::min(low, point);
        high = glm::max(high, point);
    }
    return std::max({high.x - low.x, high.y - low.y, high.z - low.z});
}

namespace {

// steak.obj is centimeters, so a long axis of 5–45 cm keeps 1 unit = 1 cm. Anything else is fit to 22 cm.
float metersPerModelUnit(const std::vector<glm::vec3>& positions){
    const float longest = longestAxis(positions);
    if (!(longest > 0.0f)) throw std::runtime_error("mesh has no extent");
    const float centimeters = longest * 0.01f;
    if (centimeters >= 0.05f && centimeters <= 0.45f) return 0.01f;
    return 0.22f / longest;
}

// A model mapped by projecting straight down squeezes each vertical wall into a line of texels. Such
// wall triangles get new coordinates: distance along the wall's outline runs across the part's UV box
// and height runs up it at the same texel density, so the texture is neither squeezed nor folded
// where the outline turns back on itself (the notch of a ribeye).
struct WallMapping{
    std::unordered_map<uint32_t, float> arc;  // source vertex -> distance along the outline, model units
    float perimeter = 0.0f;
    bool closed = false;
    float zLow = 0.0f;
    float vStart = 0.0f;
    float uStart = 0.0f;
    float uPerUnit = 0.0f;
    float vPerUnit = 0.0f;

    // A closed outline runs out over its first half and back over the second, so the texture meets
    // itself everywhere instead of at a seam.
    glm::vec2 uv(const Mesh& mesh, uint32_t vertex) const{
        const float s = arc.at(vertex);
        const float along = closed ? std::min(s, perimeter - s) : s;
        return {uStart + along * uPerUnit, vStart + (mesh.positions[vertex].z - zLow) * vPerUnit};
    }
};

float uvArea(glm::vec2 a, glm::vec2 b, glm::vec2 c){
    const glm::vec2 e1 = b - a;
    const glm::vec2 e2 = c - a;
    return 0.5f * std::abs(e1.x * e2.y - e1.y * e2.x);
}

// Per triangle of `part`: within 60 degrees of vertical, with under 5% of the texture area per model
// area that the part's flat faces get.
std::vector<bool> collapsedWalls(const Mesh& mesh, const MeshPart& part){
    const uint32_t count = part.indexCount / 3;
    std::vector<float> density(count, 0.0f);
    std::vector<bool> steep(count, false);
    std::vector<float> flatDensity;
    for (uint32_t t = 0; t < count; ++t){
        const uint32_t* tri = &mesh.indices[part.indexOffset + 3 * t];
        const glm::vec3 a = mesh.positions[tri[0]], b = mesh.positions[tri[1]], c = mesh.positions[tri[2]];
        const glm::vec3 normal = glm::cross(b - a, c - a);
        const float length = glm::length(normal);
        if (!(length > 0.0f)) continue;
        density[t] = uvArea(mesh.texcoords[tri[0]], mesh.texcoords[tri[1]], mesh.texcoords[tri[2]]) / (0.5f * length);
        steep[t] = std::abs(normal.z) < 0.5f * length;
        if (std::abs(normal.z) > 0.9f * length) flatDensity.push_back(density[t]);
    }
    std::vector<bool> walls(count, false);
    if (flatDensity.empty()) return walls;
    auto middle = flatDensity.begin() + static_cast<std::ptrdiff_t>(flatDensity.size() / 2);
    std::nth_element(flatDensity.begin(), middle, flatDensity.end());
    const float threshold = 0.05f * *middle;
    for (uint32_t t = 0; t < count; ++t) walls[t] = steep[t] && density[t] < threshold;
    return walls;
}

WallMapping wallMapping(const Mesh& mesh, const MeshPart& part, const std::vector<bool>& walls, int imageWidth,
                        int imageHeight){
    WallMapping mapping;
    glm::vec2 low(std::numeric_limits<float>::infinity());
    glm::vec2 high(-std::numeric_limits<float>::infinity());
    glm::vec2 uvLow(std::numeric_limits<float>::infinity());
    glm::vec2 uvHigh(-std::numeric_limits<float>::infinity());
    for (uint32_t n = 0; n < part.indexCount; ++n){
        const uint32_t i = mesh.indices[part.indexOffset + n];
        low = glm::min(low, glm::vec2(mesh.positions[i]));
        high = glm::max(high, glm::vec2(mesh.positions[i]));
        uvLow = glm::min(uvLow, mesh.texcoords[i]);
        uvHigh = glm::max(uvHigh, mesh.texcoords[i]);
    }

    // Wall vertices stacked at the same xy form one column of the outline.
    const float quantum = 1.0e-4f * std::max(glm::length(high - low), 1.0e-6f);
    std::map<std::pair<long, long>, int> columnOf;
    std::vector<glm::vec2> columns;
    std::unordered_map<uint32_t, int> vertexColumn;
    std::vector<std::set<int>> neighbors;
    mapping.zLow = std::numeric_limits<float>::infinity();
    float zHigh = -std::numeric_limits<float>::infinity();
    for (uint32_t t = 0; t < walls.size(); ++t){
        if (!walls[t]) continue;
        const uint32_t* tri = &mesh.indices[part.indexOffset + 3 * t];
        int ids[3];
        for (int c = 0; c < 3; ++c){
            const glm::vec3 p = mesh.positions[tri[c]];
            mapping.zLow = std::min(mapping.zLow, p.z);
            zHigh = std::max(zHigh, p.z);
            const std::pair<long, long> key{std::lround(p.x / quantum), std::lround(p.y / quantum)};
            auto [it, added] = columnOf.emplace(key, static_cast<int>(columns.size()));
            if (added){
                columns.emplace_back(p.x, p.y);
                neighbors.emplace_back();
            }
            ids[c] = it->second;
            vertexColumn[tri[c]] = it->second;
        }
        for (int c = 0; c < 3; ++c){
            const int from = ids[c];
            const int to = ids[(c + 1) % 3];
            if (from == to) continue;
            neighbors[static_cast<size_t>(from)].insert(to);
            neighbors[static_cast<size_t>(to)].insert(from);
        }
    }
    if (columns.empty()) return mapping;

    // Walk each connected outline, starting at an open end when it has one.
    std::vector<float> arc(columns.size(), 0.0f);
    std::vector<bool> visited(columns.size(), false);
    float travelled = 0.0f;
    for (size_t seed = 0; seed < columns.size(); ++seed){
        if (visited[seed]) continue;
        int start = static_cast<int>(seed);
        for (size_t c = 0; c < columns.size(); ++c){
            if (!visited[c] && neighbors[c].size() == 1){
                start = static_cast<int>(c);
                break;
            }
        }
        int current = start;
        int last = start;
        while (current >= 0){
            visited[static_cast<size_t>(current)] = true;
            travelled += glm::length(columns[static_cast<size_t>(current)] - columns[static_cast<size_t>(last)]);
            arc[static_cast<size_t>(current)] = travelled;
            last = current;
            int next = -1;
            for (int candidate : neighbors[static_cast<size_t>(current)]){
                if (!visited[static_cast<size_t>(candidate)]){
                    next = candidate;
                    break;
                }
            }
            current = next;
        }
        // A closed outline also spans the step from its last column back to the first.
        if (last != start && neighbors[static_cast<size_t>(last)].count(start)){
            travelled += glm::length(columns[static_cast<size_t>(start)] - columns[static_cast<size_t>(last)]);
            mapping.closed = true;
        }
    }
    mapping.perimeter = travelled;
    for (const auto& [vertex, column] : vertexColumn) mapping.arc[vertex] = arc[static_cast<size_t>(column)];

    // The outline spans the part's U range (half the outline when it is mirrored). The strip of the
    // wall's height sits in the middle of the V range, away from the image edges.
    const float span = (mapping.closed ? 0.5f : 1.0f) * mapping.perimeter;
    mapping.uStart = uvLow.x;
    mapping.uPerUnit = (uvHigh.x - uvLow.x) / std::max(span, 1.0e-6f);
    mapping.vPerUnit = mapping.uPerUnit * static_cast<float>(imageWidth) / static_cast<float>(imageHeight);
    const float stripHeight = (zHigh - mapping.zLow) * mapping.vPerUnit;
    mapping.vStart = std::max(uvLow.y, 0.5f * (uvLow.y + uvHigh.y - stripHeight));
    return mapping;
}

int imageForMaterial(const Mesh& mesh, int material){
    if (material < 0 || static_cast<size_t>(material) >= mesh.materials.size()){
        throw std::runtime_error("mesh part has no material");
    }
    const int image = mesh.materials[static_cast<size_t>(material)].baseColorImage;
    if (image < 0 || static_cast<size_t>(image) >= mesh.images.size()){
        throw std::runtime_error("material " + mesh.materials[static_cast<size_t>(material)].name + " has no base-color image");
    }
    return image;
}

}  // namespace

DemoMesh prepareDemoMesh(const Mesh& source, const std::string& fallbackTexture){
    const Mesh mesh = dropOverlappingParts(source);
    DemoMesh out;
    if (mesh.images.empty() || mesh.parts.empty()){
        out.positions = mesh.positions;
        out.indices = mesh.indices;
        out.solverUvs = albedoTexcoords(mesh);
        out.displayUvs = mesh.texcoords;
        if (mesh.format == MeshFormat::Gltf){
            for (glm::vec2& uv : out.displayUvs) uv.y = 1.0f - uv.y;
        }
        out.atlas = loadRgba(fallbackTexture);
        out.metersPerUnit = metersPerModelUnit(out.positions);
        out.panAxis = thinnestAxis(out.positions);
        return out;
    }

    std::vector<RgbaImage> decoded;
    decoded.reserve(mesh.images.size());
    for (const MeshImage& image : mesh.images){
        if (image.encoded.empty()) throw std::runtime_error("embedded image has no bytes: " + image.name);
        decoded.push_back(decodeMemory(image.encoded.data(), static_cast<int>(image.encoded.size()), image.name));
    }
    int atlasWidth = 0;
    int atlasHeight = 1;
    std::vector<int> originX(decoded.size(), 0);
    for (size_t i = 0; i < decoded.size(); ++i){
        originX[i] = atlasWidth;
        atlasWidth += decoded[i].width;
        atlasHeight = std::max(atlasHeight, decoded[i].height);
    }
    out.atlas = {atlasWidth, atlasHeight, std::vector<uint8_t>(static_cast<size_t>(atlasWidth) * atlasHeight * 4, 0)};
    for (size_t i = 0; i < decoded.size(); ++i){
        const RgbaImage& src = decoded[i];
        for (int y = 0; y < src.height; ++y){
            const uint8_t* from = src.pixels.data() + static_cast<size_t>(y) * src.width * 4;
            uint8_t* to = out.atlas.pixels.data() + (static_cast<size_t>(y) * atlasWidth + originX[i]) * 4;
            std::memcpy(to, from, static_cast<size_t>(src.width) * 4);
        }
    }

    size_t rewrapped = 0;
    for (const MeshPart& part : mesh.parts){
        const int image = imageForMaterial(mesh, part.material);
        const RgbaImage& src = decoded[static_cast<size_t>(image)];
        const auto addVertex = [&](glm::vec3 position, glm::vec2 uv){
            const float su = (uv.x * static_cast<float>(src.width) + static_cast<float>(originX[static_cast<size_t>(image)])) /
                             static_cast<float>(atlasWidth);
            const float sv = uv.y * static_cast<float>(src.height) / static_cast<float>(atlasHeight);
            out.positions.push_back(position);
            out.solverUvs.push_back({su, sv});
            out.displayUvs.push_back({su, 1.0f - sv});
            return static_cast<uint32_t>(out.positions.size() - 1);
        };
        if (part.indexCount % 3 != 0) throw std::runtime_error("mesh part is not a triangle list");
        for (uint32_t n = 0; n < part.indexCount; ++n){
            const uint32_t srcIndex = mesh.indices[part.indexOffset + n];
            if (srcIndex >= mesh.positions.size() || srcIndex >= mesh.texcoords.size()){
                throw std::runtime_error("mesh part index is out of range");
            }
        }
        const std::vector<bool> walls = collapsedWalls(mesh, part);
        const WallMapping wallUvs = wallMapping(mesh, part, walls, src.width, src.height);
        std::vector<int> remap(mesh.positions.size(), -1);
        for (uint32_t n = 0; n < part.indexCount; n += 3){
            const uint32_t tri[3] = {mesh.indices[part.indexOffset + n], mesh.indices[part.indexOffset + n + 1],
                                     mesh.indices[part.indexOffset + n + 2]};
            if (walls[n / 3]){
                ++rewrapped;
                for (uint32_t srcIndex : tri) out.indices.push_back(addVertex(mesh.positions[srcIndex], wallUvs.uv(mesh, srcIndex)));
                continue;
            }
            for (uint32_t srcIndex : tri){
                int& slot = remap[srcIndex];
                if (slot < 0) slot = static_cast<int>(addVertex(mesh.positions[srcIndex], mesh.texcoords[srcIndex]));
                out.indices.push_back(static_cast<uint32_t>(slot));
            }
        }
    }
    if (out.positions.empty()) throw std::runtime_error("mesh has no triangles");
    out.metersPerUnit = metersPerModelUnit(out.positions);
    out.panAxis = thinnestAxis(out.positions);
    if (source.parts.size() != mesh.parts.size()){
        std::cout << "Dropped " << source.parts.size() - mesh.parts.size() << " overlapping duplicate part"
                  << (source.parts.size() - mesh.parts.size() == 1 ? "" : "s") << "\n";
    }
    if (rewrapped > 0) std::cout << "Rewrapped texture coordinates on " << rewrapped << " side-wall triangles\n";
    std::cout << "Packed " << decoded.size() << " base-color images into a " << atlasWidth << "x" << atlasHeight << " atlas\n";
    return out;
}

HeatSurface demoSurface(const DemoMesh& mesh){
    HeatSurface surface;
    surface.positions = mesh.positions;
    surface.indices = mesh.indices;
    surface.texcoords = mesh.displayUvs;
    surface.rgba = mesh.atlas.pixels;
    surface.width = mesh.atlas.width;
    surface.height = mesh.atlas.height;
    surface.metersPerUnit = mesh.metersPerUnit;
    return surface;
}

void checkCellCount(const DemoMesh& mesh, float spacing){
    const float voxelSize = spacing / mesh.metersPerUnit;
    if (!(voxelSize > 0.0f) || longestAxis(mesh.positions) / voxelSize > 500.0f){
        throw std::invalid_argument("Cell size is too small; the grid would exceed 512 cells on a side.");
    }
}

namespace {

class DemoLive : public LiveHeat{
public:
    DemoLive(DemoMesh meshIn, const std::string& name) : mesh(std::move(meshIn)), ctx(name.c_str()){}

    HeatGrid restart(const HeatParams& params) override{
        checkCellCount(mesh, params.spacing);
        run.reset();
        run = std::make_unique<MeshHeatColorSession>(ctx, mesh.positions, mesh.solverUvs, mesh.indices, mesh.metersPerUnit, params,
                                                     mesh.atlas.pixels.data(), static_cast<uint32_t>(mesh.atlas.width),
                                                     static_cast<uint32_t>(mesh.atlas.height));
        const glm::uvec3 dims = run->voxels().dims;
        std::cout << "Voxelized into a " << dims.x << "x" << dims.y << "x" << dims.z << " grid (dx "
                  << params.spacing * 1000.0f << " mm, dt " << params.timeStep << " s, r = " << params.fourier() << ")\n";
        return run->read();
    }

    HeatGrid advance(uint32_t steps) override{
        if (!run) throw std::logic_error("advance before restart");
        run->advance(steps);
        return run->read();
    }

private:
    DemoMesh mesh;
    ComputeContext ctx;
    std::unique_ptr<MeshHeatColorSession> run;
};

}  // namespace

int runLiveDemo(const std::string& name, const std::string& path, const std::string& fallbackTexture, HeatParams params){
    try{
        DemoMesh mesh = prepareDemoMesh(loadMesh(path), fallbackTexture);
        params.panAxis = mesh.panAxis;
        std::cout << "Mesh scale: 1 unit = " << mesh.metersPerUnit * 100.0f << " cm, long axis "
                  << longestAxis(mesh.positions) * mesh.metersPerUnit * 100.0f << " cm, "
                  << (params.grillSpokes > 0 ? "grill" : "pan") << " on the low " << "xyz"[mesh.panAxis] << " side\n";
        HeatSurface surface = demoSurface(mesh);
        showLiveHeatGrid(name + ": " + path, params, std::make_unique<DemoLive>(std::move(mesh), name), true, std::move(surface));
    } catch (const std::exception& e){
        std::cerr << name << ": " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
