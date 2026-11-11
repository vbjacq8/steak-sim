#include "mesh_loader.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define TINYGLTF_IMPLEMENTATION
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#include <tiny_gltf.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace {

// Add a row here and in kExtensions when MeshFormat gains a value.
struct FormatDesc{
    MeshFormat format;
    const char* name;
    Mesh (*load)(const std::string& path);
};

struct ExtensionDesc{
    const char* extension;
    MeshFormat format;
};

Mesh loadObj(const std::string& path);
Mesh loadGltf(const std::string& path);

const FormatDesc kFormats[] = {
    {MeshFormat::Obj, "obj", loadObj},
    {MeshFormat::Gltf, "gltf", loadGltf},
};

const ExtensionDesc kExtensions[] = {
    {".obj", MeshFormat::Obj},
    {".gltf", MeshFormat::Gltf},
    {".glb", MeshFormat::Gltf},
};

const FormatDesc* describe(MeshFormat format){
    for (const FormatDesc& desc : kFormats){
        if (desc.format == format) return &desc;
    }
    return nullptr;
}

std::string lowerExtension(std::string_view path){
    std::string ext = std::filesystem::path{std::string(path)}.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext;
}

void requireTriangles(const Mesh& mesh, const std::string& path){
    if (mesh.positions.empty() || mesh.indices.size() < 3){
        throw std::runtime_error("loadMesh: " + path + " has no triangles");
    }
    if (mesh.indices.size() % 3 != 0){
        throw std::runtime_error("loadMesh: " + path + " is not a triangle list");
    }
    if (!mesh.texcoords.empty() && mesh.texcoords.size() != mesh.positions.size()){
        throw std::runtime_error("loadMesh: texcoord count does not match the vertex count");
    }
    if (!mesh.normals.empty() && mesh.normals.size() != mesh.positions.size()){
        throw std::runtime_error("loadMesh: normal count does not match the vertex count");
    }
    for (uint32_t index : mesh.indices){
        if (index >= mesh.positions.size()) throw std::runtime_error("loadMesh: index out of range in " + path);
    }
}

glm::vec2 objTexcoord(const tinyobj::attrib_t& attrib, int texcoordIndex){
    const size_t i = static_cast<size_t>(texcoordIndex) * 2;
    if (texcoordIndex < 0 || i + 1 >= attrib.texcoords.size()){
        throw std::runtime_error("loadMesh: obj texcoord index is out of range");
    }
    return {attrib.texcoords[i], attrib.texcoords[i + 1]};
}

glm::vec3 objNormal(const tinyobj::attrib_t& attrib, int normalIndex){
    const size_t i = static_cast<size_t>(normalIndex) * 3;
    if (normalIndex < 0 || i + 2 >= attrib.normals.size()){
        throw std::runtime_error("loadMesh: obj normal index is out of range");
    }
    return {attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2]};
}

struct CornerKey{
    int vertex = 0;
    int texcoord = 0;
    int normal = 0;
    bool operator==(const CornerKey&) const = default;
};

struct CornerHash{
    size_t operator()(const CornerKey& key) const{
        size_t h = std::hash<int>{}(key.vertex);
        h ^= std::hash<int>{}(key.texcoord) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(key.normal) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Positions stay in the OBJ vertex order so unreferenced vertices still contribute to the bounds.
// A corner is split only when the same position index is reused with a different texcoord or normal.
Mesh loadObj(const std::string& path){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())){
        const std::string message = warn + err;
        throw std::runtime_error(message.empty() ? "loadMesh: failed to read " + path : message);
    }

    Mesh mesh;
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3){
        mesh.positions.push_back({attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]});
    }
    const size_t vertexCount = mesh.positions.size();
    const bool trackTex = !attrib.texcoords.empty();
    const bool trackNormal = !attrib.normals.empty();
    if (trackTex) mesh.texcoords.assign(vertexCount, {});
    if (trackNormal) mesh.normals.assign(vertexCount, {});

    std::vector<char> texBound(vertexCount, 0);
    std::vector<char> normalBound(vertexCount, 0);
    std::vector<int> boundTex(vertexCount, -1);
    std::vector<int> boundNormal(vertexCount, -1);
    std::unordered_map<CornerKey, uint32_t, CornerHash> splits;

    const auto corner = [&](const tinyobj::index_t& index){
        if (index.vertex_index < 0 || static_cast<size_t>(index.vertex_index) >= vertexCount){
            throw std::runtime_error("loadMesh: obj vertex index is out of range");
        }
        const int vertex = index.vertex_index;
        const int texcoord = index.texcoord_index;
        const int normal = index.normal_index;
        const glm::vec2 uv = trackTex && texcoord >= 0 ? objTexcoord(attrib, texcoord) : glm::vec2(0.0f);
        const glm::vec3 nrm = trackNormal && normal >= 0 ? objNormal(attrib, normal) : glm::vec3(0.0f);

        const bool newTex = trackTex && !texBound[vertex];
        const bool newNormal = trackNormal && !normalBound[vertex];
        if (newTex){
            texBound[vertex] = 1;
            boundTex[vertex] = texcoord;
            mesh.texcoords[static_cast<size_t>(vertex)] = uv;
        }
        if (newNormal){
            normalBound[vertex] = 1;
            boundNormal[vertex] = normal;
            mesh.normals[static_cast<size_t>(vertex)] = nrm;
        }
        const bool texConflict = trackTex && !newTex && boundTex[vertex] != texcoord;
        const bool normalConflict = trackNormal && !newNormal && boundNormal[vertex] != normal;
        if (!texConflict && !normalConflict){
            mesh.indices.push_back(static_cast<uint32_t>(vertex));
            return;
        }

        const CornerKey key{vertex, texcoord, normal};
        const auto found = splits.find(key);
        if (found != splits.end()){
            mesh.indices.push_back(found->second);
            return;
        }
        if (mesh.positions.size() > std::numeric_limits<uint32_t>::max()){
            throw std::runtime_error("loadMesh: too many vertices in " + path);
        }
        const auto id = static_cast<uint32_t>(mesh.positions.size());
        const glm::vec3 position = mesh.positions[static_cast<size_t>(vertex)];
        mesh.positions.push_back(position);
        if (trackTex) mesh.texcoords.push_back(uv);
        if (trackNormal) mesh.normals.push_back(nrm);
        splits.emplace(key, id);
        mesh.indices.push_back(id);
    };

    for (const tinyobj::shape_t& shape : shapes){
        if (shape.mesh.num_face_vertices.empty()){
            if (shape.mesh.indices.size() % 3 != 0){
                throw std::runtime_error("loadMesh: obj face is not a triangle in " + path);
            }
            for (const tinyobj::index_t& index : shape.mesh.indices) corner(index);
            continue;
        }
        size_t offset = 0;
        for (unsigned faceVerts : shape.mesh.num_face_vertices){
            if (faceVerts != 3 || offset + faceVerts > shape.mesh.indices.size()){
                throw std::runtime_error("loadMesh: obj face is not a triangle in " + path);
            }
            corner(shape.mesh.indices[offset]);
            corner(shape.mesh.indices[offset + 1]);
            corner(shape.mesh.indices[offset + 2]);
            offset += faceVerts;
        }
    }
    return mesh;
}

struct Packed{
    const unsigned char* data = nullptr;
    size_t stride = 0;
    size_t count = 0;
    int componentType = 0;
    bool normalized = false;
};

Packed accessorPacked(const tinygltf::Model& model, int index, int expectType){
    if (index < 0 || static_cast<size_t>(index) >= model.accessors.size()){
        throw std::runtime_error("loadMesh: glTF accessor index is out of range");
    }
    const tinygltf::Accessor& accessor = model.accessors[static_cast<size_t>(index)];
    if (accessor.sparse.isSparse) throw std::runtime_error("loadMesh: sparse glTF accessors are not supported");
    if (accessor.type != expectType) throw std::runtime_error("loadMesh: glTF accessor has an unexpected type");
    if (accessor.bufferView < 0 || static_cast<size_t>(accessor.bufferView) >= model.bufferViews.size()){
        throw std::runtime_error("loadMesh: glTF accessor is missing a buffer view");
    }
    const tinygltf::BufferView& view = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    if (view.buffer < 0 || static_cast<size_t>(view.buffer) >= model.buffers.size()){
        throw std::runtime_error("loadMesh: glTF buffer view is missing a buffer");
    }
    const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(view.buffer)];
    const int compSize = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(accessor.componentType));
    const int components = tinygltf::GetNumComponentsInType(static_cast<uint32_t>(accessor.type));
    if (compSize <= 0 || components <= 0){
        throw std::runtime_error("loadMesh: glTF accessor has an unsupported component type");
    }
    const int tight = compSize * components;
    const int stride = accessor.ByteStride(view);
    if (stride < tight) throw std::runtime_error("loadMesh: glTF accessor stride is invalid");

    const size_t start = view.byteOffset + accessor.byteOffset;
    if (accessor.count > 0){
        const size_t span = accessor.count - 1;
        const auto step = static_cast<size_t>(stride);
        const auto elem = static_cast<size_t>(tight);
        if (start > buffer.data.size() || (step != 0 && span > (std::numeric_limits<size_t>::max() - elem) / step)){
            throw std::runtime_error("loadMesh: glTF accessor runs past the end of its buffer");
        }
        const size_t end = start + span * step + elem;
        if (end < start || end > buffer.data.size()){
            throw std::runtime_error("loadMesh: glTF accessor runs past the end of its buffer");
        }
    }
    return {buffer.data.data() + start, static_cast<size_t>(stride), accessor.count, accessor.componentType, accessor.normalized};
}

float componentAsFloat(const unsigned char* p, int componentType, bool normalized){
    switch (componentType){
        case TINYGLTF_COMPONENT_TYPE_FLOAT:{
            float v;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case TINYGLTF_COMPONENT_TYPE_DOUBLE:{
            double v;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:{
            const auto v = *p;
            return normalized ? static_cast<float>(v) / 255.0f : static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_BYTE:{
            int8_t v;
            std::memcpy(&v, p, sizeof(v));
            return normalized ? std::max(static_cast<float>(v) / 127.0f, -1.0f) : static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:{
            uint16_t v;
            std::memcpy(&v, p, sizeof(v));
            return normalized ? static_cast<float>(v) / 65535.0f : static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_SHORT:{
            int16_t v;
            std::memcpy(&v, p, sizeof(v));
            return normalized ? std::max(static_cast<float>(v) / 32767.0f, -1.0f) : static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:{
            uint32_t v;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<float>(v);
        }
        case TINYGLTF_COMPONENT_TYPE_INT:{
            int32_t v;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<float>(v);
        }
        default: throw std::runtime_error("loadMesh: glTF accessor has an unsupported component type");
    }
}

uint32_t componentAsIndex(const unsigned char* p, int componentType){
    switch (componentType){
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: return *p;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:{
            uint16_t v;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:{
            uint32_t v;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        default: throw std::runtime_error("loadMesh: glTF indices must be unsigned integers");
    }
}

std::vector<glm::vec3> readVec3(const tinygltf::Model& model, int accessor){
    const Packed packed = accessorPacked(model, accessor, TINYGLTF_TYPE_VEC3);
    const int compSize = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(packed.componentType));
    std::vector<glm::vec3> out(packed.count);
    for (size_t i = 0; i < packed.count; ++i){
        const unsigned char* p = packed.data + i * packed.stride;
        out[i] = {
            componentAsFloat(p, packed.componentType, packed.normalized),
            componentAsFloat(p + compSize, packed.componentType, packed.normalized),
            componentAsFloat(p + 2 * compSize, packed.componentType, packed.normalized),
        };
    }
    return out;
}

std::vector<glm::vec2> readVec2(const tinygltf::Model& model, int accessor){
    const Packed packed = accessorPacked(model, accessor, TINYGLTF_TYPE_VEC2);
    const int compSize = tinygltf::GetComponentSizeInBytes(static_cast<uint32_t>(packed.componentType));
    std::vector<glm::vec2> out(packed.count);
    for (size_t i = 0; i < packed.count; ++i){
        const unsigned char* p = packed.data + i * packed.stride;
        out[i] = {
            componentAsFloat(p, packed.componentType, packed.normalized),
            componentAsFloat(p + compSize, packed.componentType, packed.normalized),
        };
    }
    return out;
}

std::vector<uint32_t> readIndices(const tinygltf::Model& model, int accessor){
    const Packed packed = accessorPacked(model, accessor, TINYGLTF_TYPE_SCALAR);
    std::vector<uint32_t> out(packed.count);
    for (size_t i = 0; i < packed.count; ++i){
        out[i] = componentAsIndex(packed.data + i * packed.stride, packed.componentType);
    }
    return out;
}

void appendTriangles(std::vector<uint32_t>& indices, const std::vector<uint32_t>& corners, int mode, uint32_t base,
                     uint32_t vertexCount){
    if (vertexCount > 0 && base > std::numeric_limits<uint32_t>::max() - (vertexCount - 1)){
        throw std::runtime_error("loadMesh: too many vertices");
    }
    const auto emit = [&](uint32_t a, uint32_t b, uint32_t c){
        if (a >= vertexCount || b >= vertexCount || c >= vertexCount){
            throw std::runtime_error("loadMesh: glTF index is out of range");
        }
        indices.push_back(base + a);
        indices.push_back(base + b);
        indices.push_back(base + c);
    };
    if (mode == TINYGLTF_MODE_TRIANGLES){
        if (corners.size() % 3 != 0) throw std::runtime_error("loadMesh: glTF triangle index count is not a multiple of 3");
        for (size_t i = 0; i < corners.size(); i += 3) emit(corners[i], corners[i + 1], corners[i + 2]);
    } else if (mode == TINYGLTF_MODE_TRIANGLE_STRIP){
        for (size_t i = 0; i + 2 < corners.size(); ++i){
            if ((i % 2) == 0) emit(corners[i], corners[i + 1], corners[i + 2]);
            else emit(corners[i + 1], corners[i], corners[i + 2]);
        }
    } else if (mode == TINYGLTF_MODE_TRIANGLE_FAN){
        for (size_t i = 1; i + 1 < corners.size(); ++i) emit(corners[0], corners[i], corners[i + 1]);
    }
}

glm::mat4 nodeMatrix(const tinygltf::Node& node){
    if (node.matrix.size() == 16){
        float m[16];
        for (int i = 0; i < 16; ++i) m[i] = static_cast<float>(node.matrix[static_cast<size_t>(i)]);
        return glm::make_mat4(m);
    }
    glm::vec3 translation(0.0f);
    glm::vec3 scale(1.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
    if (node.translation.size() == 3){
        translation = {static_cast<float>(node.translation[0]), static_cast<float>(node.translation[1]),
                       static_cast<float>(node.translation[2])};
    }
    if (node.scale.size() == 3){
        scale = {static_cast<float>(node.scale[0]), static_cast<float>(node.scale[1]), static_cast<float>(node.scale[2])};
    }
    if (node.rotation.size() == 4){
        // glTF stores quaternions as xyzw. glm::quat takes wxyz.
        rotation = glm::quat(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                             static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]));
    }
    return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

glm::mat3 normalMatrix(const glm::mat4& xform){
    const glm::mat3 linear(xform);
    if (std::abs(glm::determinant(linear)) < 1e-12f) return linear;
    return glm::transpose(glm::inverse(linear));
}

int attributeIndex(const tinygltf::Primitive& prim, const char* name){
    const auto it = prim.attributes.find(name);
    return it == prim.attributes.end() ? -1 : it->second;
}

void appendPrimitive(Mesh& mesh, const tinygltf::Model& model, const tinygltf::Primitive& prim, const glm::mat4& xform,
                     const std::string& name){
    if (prim.extensions.find("KHR_draco_mesh_compression") != prim.extensions.end()){
        throw std::runtime_error("loadMesh: Draco-compressed glTF is not supported");
    }
    const int mode = prim.mode < 0 ? TINYGLTF_MODE_TRIANGLES : prim.mode;
    if (mode != TINYGLTF_MODE_TRIANGLES && mode != TINYGLTF_MODE_TRIANGLE_STRIP && mode != TINYGLTF_MODE_TRIANGLE_FAN) return;

    const int positionAccessor = attributeIndex(prim, "POSITION");
    if (positionAccessor < 0) throw std::runtime_error("loadMesh: glTF primitive is missing POSITION");
    const std::vector<glm::vec3> positions = readVec3(model, positionAccessor);
    if (positions.empty()) return;
    if (positions.size() > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("loadMesh: too many vertices");

    const int uvAccessor = attributeIndex(prim, "TEXCOORD_0");
    const int normalAccessor = attributeIndex(prim, "NORMAL");
    std::vector<glm::vec2> uvs = uvAccessor >= 0 ? readVec2(model, uvAccessor) : std::vector<glm::vec2>{};
    std::vector<glm::vec3> normals = normalAccessor >= 0 ? readVec3(model, normalAccessor) : std::vector<glm::vec3>{};
    if (!uvs.empty() && uvs.size() != positions.size()){
        throw std::runtime_error("loadMesh: glTF texcoord count does not match POSITION");
    }
    if (!normals.empty() && normals.size() != positions.size()){
        throw std::runtime_error("loadMesh: glTF normal count does not match POSITION");
    }

    if (mesh.positions.size() > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("loadMesh: too many vertices");
    const auto base = static_cast<uint32_t>(mesh.positions.size());
    const bool wantUv = !mesh.texcoords.empty() || !uvs.empty();
    const bool wantNormal = !mesh.normals.empty() || !normals.empty();
    if (wantUv && mesh.texcoords.empty()) mesh.texcoords.assign(base, {});
    if (wantNormal && mesh.normals.empty()) mesh.normals.assign(base, {});

    const glm::mat3 nmat = normalMatrix(xform);
    for (const glm::vec3& position : positions){
        mesh.positions.push_back(glm::vec3(xform * glm::vec4(position, 1.0f)));
    }
    if (wantUv){
        if (uvs.empty()) mesh.texcoords.insert(mesh.texcoords.end(), positions.size(), {});
        else mesh.texcoords.insert(mesh.texcoords.end(), uvs.begin(), uvs.end());
    }
    if (wantNormal){
        if (normals.empty()){
            mesh.normals.insert(mesh.normals.end(), positions.size(), {});
        } else{
            for (glm::vec3 normal : normals){
                normal = nmat * normal;
                const float len = glm::length(normal);
                mesh.normals.push_back(len > 0.0f ? normal / len : glm::vec3(0.0f));
            }
        }
    }

    std::vector<uint32_t> corners;
    if (prim.indices >= 0) corners = readIndices(model, prim.indices);
    else{
        corners.resize(positions.size());
        for (uint32_t i = 0; i < corners.size(); ++i) corners[i] = i;
    }
    const auto indexOffset = static_cast<uint32_t>(mesh.indices.size());
    appendTriangles(mesh.indices, corners, mode, base, static_cast<uint32_t>(positions.size()));
    if (mesh.indices.size() == indexOffset) return;
    MeshPart part;
    part.name = name;
    part.vertexOffset = base;
    part.vertexCount = static_cast<uint32_t>(positions.size());
    part.indexOffset = indexOffset;
    part.indexCount = static_cast<uint32_t>(mesh.indices.size() - indexOffset);
    part.material = prim.material;
    mesh.parts.push_back(part);
}

void appendMesh(Mesh& mesh, const tinygltf::Model& model, const tinygltf::Mesh& gltfMesh, const glm::mat4& xform,
                const std::string& name){
    const std::string& partName = !name.empty() ? name : gltfMesh.name;
    for (const tinygltf::Primitive& prim : gltfMesh.primitives) appendPrimitive(mesh, model, prim, xform, partName);
}

void appendNode(Mesh& mesh, const tinygltf::Model& model, int nodeIndex, const glm::mat4& parent, std::vector<int>& stack){
    if (nodeIndex < 0 || static_cast<size_t>(nodeIndex) >= model.nodes.size()){
        throw std::runtime_error("loadMesh: glTF node index is out of range");
    }
    if (std::find(stack.begin(), stack.end(), nodeIndex) != stack.end()){
        throw std::runtime_error("loadMesh: glTF node hierarchy has a cycle");
    }
    stack.push_back(nodeIndex);
    const tinygltf::Node& node = model.nodes[static_cast<size_t>(nodeIndex)];
    const glm::mat4 world = parent * nodeMatrix(node);
    if (node.mesh >= 0){
        if (static_cast<size_t>(node.mesh) >= model.meshes.size()){
            throw std::runtime_error("loadMesh: glTF mesh index is out of range");
        }
        // Skinning and morph targets stay in the node's bind pose.
        appendMesh(mesh, model, model.meshes[static_cast<size_t>(node.mesh)], world, node.name);
    }
    for (int child : node.children) appendNode(mesh, model, child, world, stack);
    stack.pop_back();
}

bool fileStartsWithGlb(const std::string& path){
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    in.read(magic, 4);
    return in.gcount() == 4 && std::memcmp(magic, "glTF", 4) == 0;
}

Mesh loadGltf(const std::string& path){
    tinygltf::TinyGLTF loader;
    // Keep the PNG/JPEG bytes in the buffer. Pixels are decoded by the caller.
    loader.SetImageLoader(
        [](tinygltf::Image*, int, std::string*, std::string*, int, int, const unsigned char*, int, void*){ return true; },
        nullptr);

    tinygltf::Model model;
    std::string err, warn;
    const bool binary = fileStartsWithGlb(path);
    bool ok = false;
    try{
        ok = binary ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                    : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    } catch (const std::exception& e){
        throw std::runtime_error(std::string("loadMesh: ") + e.what());
    }
    if (!ok){
        const std::string message = warn + err;
        throw std::runtime_error(message.empty() ? "loadMesh: failed to read " + path : message);
    }

    Mesh mesh;
    std::vector<int> stack;
    if (!model.scenes.empty()){
        const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
        if (sceneIndex < 0 || static_cast<size_t>(sceneIndex) >= model.scenes.size()){
            throw std::runtime_error("loadMesh: glTF scene index is out of range");
        }
        for (int root : model.scenes[static_cast<size_t>(sceneIndex)].nodes){
            appendNode(mesh, model, root, glm::mat4(1.0f), stack);
        }
    } else if (!model.nodes.empty()){
        std::vector<char> isChild(model.nodes.size(), 0);
        for (const tinygltf::Node& node : model.nodes){
            for (int child : node.children){
                if (child >= 0 && static_cast<size_t>(child) < isChild.size()) isChild[static_cast<size_t>(child)] = 1;
            }
        }
        for (int i = 0; i < static_cast<int>(model.nodes.size()); ++i){
            if (!isChild[static_cast<size_t>(i)]) appendNode(mesh, model, i, glm::mat4(1.0f), stack);
        }
    } else{
        for (const tinygltf::Mesh& gltfMesh : model.meshes) appendMesh(mesh, model, gltfMesh, glm::mat4(1.0f), gltfMesh.name);
    }

    for (const tinygltf::Image& image : model.images){
        MeshImage out;
        out.name = image.name;
        out.mimeType = image.mimeType;
        if (image.bufferView >= 0 && static_cast<size_t>(image.bufferView) < model.bufferViews.size()){
            const tinygltf::BufferView& view = model.bufferViews[static_cast<size_t>(image.bufferView)];
            if (view.buffer >= 0 && static_cast<size_t>(view.buffer) < model.buffers.size()){
                const tinygltf::Buffer& buffer = model.buffers[static_cast<size_t>(view.buffer)];
                const size_t begin = view.byteOffset;
                const size_t end = begin + view.byteLength;
                if (end <= buffer.data.size()) out.encoded.assign(buffer.data.begin() + begin, buffer.data.begin() + end);
            }
        }
        mesh.images.push_back(std::move(out));
    }
    for (const tinygltf::Material& material : model.materials){
        MeshMaterial out;
        out.name = material.name;
        const int texture = material.pbrMetallicRoughness.baseColorTexture.index;
        if (texture >= 0 && static_cast<size_t>(texture) < model.textures.size()){
            out.baseColorImage = model.textures[static_cast<size_t>(texture)].source;
        }
        mesh.materials.push_back(out);
    }
    return mesh;
}

}  // namespace

namespace {

float partVolume(glm::vec3 low, glm::vec3 high){
    const glm::vec3 extent = glm::max(high - low, glm::vec3(0.0f));
    return extent.x * extent.y * extent.z;
}

float overlapRatio(glm::vec3 aLow, glm::vec3 aHigh, glm::vec3 bLow, glm::vec3 bHigh){
    const float smaller = std::min(partVolume(aLow, aHigh), partVolume(bLow, bHigh));
    if (!(smaller > 0.0f)) return 0.0f;
    const glm::vec3 extent = glm::max(glm::min(aHigh, bHigh) - glm::max(aLow, bLow), glm::vec3(0.0f));
    return extent.x * extent.y * extent.z / smaller;
}

void partBounds(const Mesh& mesh, const MeshPart& part, glm::vec3& low, glm::vec3& high){
    low = glm::vec3(std::numeric_limits<float>::infinity());
    high = glm::vec3(-std::numeric_limits<float>::infinity());
    for (uint32_t i = 0; i < part.indexCount; ++i){
        const uint32_t index = mesh.indices[part.indexOffset + i];
        if (index >= mesh.positions.size()) continue;
        low = glm::min(low, mesh.positions[index]);
        high = glm::max(high, mesh.positions[index]);
    }
}

}  // namespace

Mesh dropOverlappingParts(Mesh mesh){
    if (mesh.parts.size() < 2) return mesh;
    std::vector<glm::vec3> low(mesh.parts.size());
    std::vector<glm::vec3> high(mesh.parts.size());
    for (size_t i = 0; i < mesh.parts.size(); ++i) partBounds(mesh, mesh.parts[i], low[i], high[i]);

    std::vector<char> keep(mesh.parts.size(), 1);
    for (size_t i = 0; i < mesh.parts.size(); ++i){
        for (size_t earlier = 0; earlier < i; ++earlier){
            if (!keep[earlier] || mesh.parts[i].material != mesh.parts[earlier].material) continue;
            if (overlapRatio(low[i], high[i], low[earlier], high[earlier]) > 0.9f){
                keep[i] = 0;
                break;
            }
        }
    }
    if (std::find(keep.begin(), keep.end(), char(0)) == keep.end()) return mesh;

    Mesh kept = mesh;
    kept.positions.clear();
    kept.normals.clear();
    kept.texcoords.clear();
    kept.indices.clear();
    kept.parts.clear();
    const bool hasUv = !mesh.texcoords.empty();
    const bool hasNormal = !mesh.normals.empty();
    std::vector<int> remap(mesh.positions.size(), -1);
    for (size_t partIndex = 0; partIndex < mesh.parts.size(); ++partIndex){
        if (!keep[partIndex]) continue;
        const MeshPart& source = mesh.parts[partIndex];
        MeshPart part = source;
        part.vertexOffset = static_cast<uint32_t>(kept.positions.size());
        part.indexOffset = static_cast<uint32_t>(kept.indices.size());
        part.indexCount = 0;
        std::fill(remap.begin(), remap.end(), -1);
        for (uint32_t i = 0; i < source.indexCount; ++i){
            const uint32_t index = mesh.indices[source.indexOffset + i];
            if (index >= mesh.positions.size()) throw std::runtime_error("loadMesh: part index is out of range");
            int& slot = remap[index];
            if (slot < 0){
                slot = static_cast<int>(kept.positions.size());
                kept.positions.push_back(mesh.positions[index]);
                if (hasUv) kept.texcoords.push_back(index < mesh.texcoords.size() ? mesh.texcoords[index] : glm::vec2(0.0f));
                if (hasNormal) kept.normals.push_back(index < mesh.normals.size() ? mesh.normals[index] : glm::vec3(0.0f));
            }
            kept.indices.push_back(static_cast<uint32_t>(slot));
            ++part.indexCount;
        }
        part.vertexCount = static_cast<uint32_t>(kept.positions.size() - part.vertexOffset);
        kept.parts.push_back(part);
    }
    return kept;
}

std::string_view meshFormatName(MeshFormat format){
    const FormatDesc* desc = describe(format);
    if (!desc) throw std::invalid_argument("loadMesh: unknown mesh format");
    return desc->name;
}

MeshFormat meshFormatFromPath(std::string_view path){
    const std::string ext = lowerExtension(path);
    for (const ExtensionDesc& desc : kExtensions){
        if (ext == desc.extension) return desc.format;
    }
    throw std::runtime_error("loadMesh: unsupported mesh extension \"" + ext + "\" for " + std::string(path));
}

Mesh loadMesh(const std::string& path, MeshFormat format){
    const FormatDesc* desc = describe(format);
    if (!desc) throw std::invalid_argument("loadMesh: unknown mesh format");
    Mesh mesh = desc->load(path);
    mesh.format = format;
    requireTriangles(mesh, path);
    return mesh;
}

Mesh loadMesh(const std::string& path){
    return loadMesh(path, meshFormatFromPath(path));
}
