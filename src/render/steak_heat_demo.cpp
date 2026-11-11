// Steak heat demo: steak mesh -> GPU voxelizer -> GPU heat solver -> cooked color -> block view.
//
// The mesh is in centimeters and is voxelized with cells HeatParams::spacing wide. Every voxel the steak
// touches conducts heat. The empty z = 0 layer is the pan, and the air around it convects at
// HeatParams::airTransfer. After each heat step, a* of the 3D color texture moves toward the paper
// equilibrium at that voxel's temperature. The window starts on the temperature field; C switches solid
// cells to that cooked color so the two can be compared. The pan stays on the temperature key.
// V switches from the voxel blocks to the textured mesh. Cooked color on that mesh samples the voxels.
// Interior voxels take the UV of the nearest surface point, so the crust texture bleeds inward.
// See heat_view.hpp for the viewer keys.
//
// Usage: steak_heat_demo [mesh.obj|mesh.gltf|mesh.glb]
//
// A glTF with several base-color images, such as src/assets/steak2/*.glb, packs those images into one
// atlas so the fat cap and the lean each keep their own texture. A model that is not already about
// the size of a steak (its long axis, read as centimeters, outside 5–45 cm) is scaled so that axis is
// 22 cm. Overlapping duplicate shells are dropped.

#include "heat_view.hpp"
#include "mesh_loader.hpp"
#include "steak_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {

struct RgbaImage{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

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

float longestAxis(const std::vector<glm::vec3>& positions){
    glm::vec3 low(std::numeric_limits<float>::infinity());
    glm::vec3 high(-std::numeric_limits<float>::infinity());
    for (const glm::vec3& point : positions){
        low = glm::min(low, point);
        high = glm::max(high, point);
    }
    return std::max({high.x - low.x, high.y - low.y, high.z - low.z});
}

// steak.obj is centimeters, so a long axis of 5–45 cm keeps 1 unit = 1 cm. Anything else is fit to 22 cm.
float metersPerModelUnit(const std::vector<glm::vec3>& positions){
    const float longest = longestAxis(positions);
    if (!(longest > 0.0f)) throw std::runtime_error("mesh has no extent");
    const float centimeters = longest * 0.01f;
    if (centimeters >= 0.05f && centimeters <= 0.45f) return 0.01f;
    return 0.22f / longest;
}

struct DemoMesh{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> solverUvs;
    std::vector<glm::vec2> displayUvs;
    std::vector<uint32_t> indices;
    RgbaImage atlas;
    float metersPerUnit = 0.01f;
};

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

    for (const MeshPart& part : mesh.parts){
        const int image = imageForMaterial(mesh, part.material);
        const RgbaImage& src = decoded[static_cast<size_t>(image)];
        std::vector<int> remap(mesh.positions.size(), -1);
        for (uint32_t n = 0; n < part.indexCount; ++n){
            const uint32_t srcIndex = mesh.indices[part.indexOffset + n];
            if (srcIndex >= mesh.positions.size() || srcIndex >= mesh.texcoords.size()){
                throw std::runtime_error("mesh part index is out of range");
            }
            int& slot = remap[srcIndex];
            if (slot < 0){
                slot = static_cast<int>(out.positions.size());
                out.positions.push_back(mesh.positions[srcIndex]);
                const glm::vec2 uv = mesh.texcoords[srcIndex];
                const float su = (uv.x * static_cast<float>(src.width) + static_cast<float>(originX[static_cast<size_t>(image)])) /
                                 static_cast<float>(atlasWidth);
                const float sv = uv.y * static_cast<float>(src.height) / static_cast<float>(atlasHeight);
                out.solverUvs.push_back({su, sv});
                out.displayUvs.push_back({su, 1.0f - sv});
            }
            out.indices.push_back(static_cast<uint32_t>(slot));
        }
    }
    if (out.positions.empty()) throw std::runtime_error("mesh has no triangles");
    out.metersPerUnit = metersPerModelUnit(out.positions);
    if (source.parts.size() != mesh.parts.size()){
        std::cout << "Dropped " << source.parts.size() - mesh.parts.size() << " overlapping duplicate part"
                  << (source.parts.size() - mesh.parts.size() == 1 ? "" : "s") << "\n";
    }
    std::cout << "Packed " << decoded.size() << " base-color images into a " << atlasWidth << "x" << atlasHeight << " atlas\n";
    return out;
}

}  // namespace

int main(int argc, char** argv){
    const std::string objPath = argc > 1 ? argv[1] : STEAK_OBJ;
    const HeatParams params;
    const auto steps = static_cast<uint32_t>(std::lround(params.duration / params.timeStep));

    try{
        const Mesh loaded = loadMesh(objPath);
        const DemoMesh mesh = prepareDemoMesh(loaded, STEAK_TEXTURE);
        std::cout << "Mesh scale: 1 unit = " << mesh.metersPerUnit * 100.0f << " cm, long axis "
                  << longestAxis(mesh.positions) * mesh.metersPerUnit * 100.0f << " cm\n";
        SteakHeatResult result;
        {
            ComputeContext ctx("steak_heat_demo");
            const auto start = std::chrono::steady_clock::now();
            result = simulateMeshHeatColor(ctx, mesh.positions, mesh.solverUvs, mesh.indices, mesh.metersPerUnit, params, steps,
                                           mesh.atlas.pixels.data(), static_cast<uint32_t>(mesh.atlas.width),
                                           static_cast<uint32_t>(mesh.atlas.height));
            const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;
            const HeatGrid& heat = result.heat;
            std::cout << "Voxelized " << mesh.indices.size() / 3 << " triangles into a " << heat.dims.x << "x" << heat.dims.y
                      << "x" << heat.dims.z << " grid (dx " << heat.spacing * 1000.0f << " mm), then ran " << steps
                      << " heat and color steps (dt " << params.timeStep << " s, r = " << params.fourier() << ") in "
                      << elapsed.count() << " ms\n";
        }
        printCenterColumn(result.heat, result.heat.dims.y);
        HeatSurface surface;
        surface.positions = mesh.positions;
        surface.indices = mesh.indices;
        surface.texcoords = mesh.displayUvs;
        surface.rgba = mesh.atlas.pixels;
        surface.width = mesh.atlas.width;
        surface.height = mesh.atlas.height;
        surface.metersPerUnit = mesh.metersPerUnit;
        const std::vector<glm::vec3> positions = mesh.positions;
        const std::vector<glm::vec2> uvs = mesh.solverUvs;
        const std::vector<uint32_t> indices = mesh.indices;
        const std::vector<uint8_t> pixels = mesh.atlas.pixels;
        const auto width = static_cast<uint32_t>(mesh.atlas.width);
        const auto height = static_cast<uint32_t>(mesh.atlas.height);
        const float metersPerUnit = mesh.metersPerUnit;
        showHeatGrid(
            "steak heat demo: " + objPath, std::move(result.heat), params, steps,
            [positions, uvs, indices, pixels, width, height, metersPerUnit](const HeatParams& next, uint32_t nextSteps,
                                                             const std::function<void(float)>& progress){
                glm::vec3 low(std::numeric_limits<float>::infinity());
                glm::vec3 high(-std::numeric_limits<float>::infinity());
                for (const glm::vec3& point : positions){
                    low = glm::min(low, point);
                    high = glm::max(high, point);
                }
                const float voxelSize = next.spacing / metersPerUnit;
                const float extent = std::max({high.x - low.x, high.y - low.y, high.z - low.z});
                if (!(voxelSize > 0.0f) || extent / voxelSize > 500.0f){
                    throw std::invalid_argument("Cell size is too small; the grid would exceed 512 cells on a side.");
                }
                ComputeContext ctx("steak_heat_demo");
                return simulateMeshHeatColor(ctx, positions, uvs, indices, metersPerUnit, next, nextSteps, pixels.data(), width,
                                             height, progress)
                    .heat;
            },
            true, std::move(surface));
    } catch (const std::exception& e){
        std::cerr << "steak_heat_demo: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
