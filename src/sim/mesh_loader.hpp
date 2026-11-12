#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

enum class MeshFormat{
    Obj,
    Gltf,
};

// One glTF primitive after its node transform. `indices` still index the parent Mesh and stay
// inside [vertexOffset, vertexOffset + vertexCount).
struct MeshPart{
    std::string name;
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    int material = -1;
};

// Embedded image. `encoded` is the file bytes (PNG or JPEG), not decoded pixels.
struct MeshImage{
    std::string name;
    std::string mimeType;
    std::vector<uint8_t> encoded;
};

struct MeshMaterial{
    std::string name;
    int baseColorImage = -1;  // index into Mesh::images, or -1
};

struct Mesh{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;    // empty, or one per position
    std::vector<glm::vec2> texcoords;  // empty, or one per position, as stored in the file
    std::vector<uint32_t> indices;     // triangle list
    MeshFormat format = MeshFormat::Obj;
    std::vector<MeshPart> parts;          // empty for OBJ; one entry per glTF primitive
    std::vector<MeshImage> images;        // embedded glTF images
    std::vector<MeshMaterial> materials;
};

// Picks the loader from the extension, case-insensitively: .obj, or .gltf / .glb.
Mesh loadMesh(const std::string& path);

// Drops a later part whose bounds cover at least 90% of an earlier part with the same material.
// glTF morph exports often include two copies of the same shell. Parts are empty for OBJ, which is unchanged.
Mesh dropOverlappingParts(Mesh mesh);
