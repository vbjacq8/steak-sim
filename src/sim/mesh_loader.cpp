#include "mesh_loader.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <tiny_gltf_v3.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace {

[[noreturn]] void fail(const std::string& message){
    throw std::runtime_error("loadMesh: " + message);
}

// Positions stay in the OBJ vertex order so unreferenced vertices still contribute to the bounds.
// The first corner to use a position claims its slot; a later corner with a different texcoord or normal gets a copy.
Mesh loadObj(const std::string& path){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.c_str())) fail(path + ": " + warn + err);

    Mesh mesh{.format = MeshFormat::Obj};
    for (size_t i = 0; i + 2 < attrib.vertices.size(); i += 3){
        mesh.positions.push_back({attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]});
    }
    const size_t vertexCount = mesh.positions.size();
    const bool hasUv = !attrib.texcoords.empty();
    const bool hasNormal = !attrib.normals.empty();
    if (hasUv) mesh.texcoords.resize(vertexCount);
    if (hasNormal) mesh.normals.resize(vertexCount);

    std::vector<char> claimed(vertexCount, 0);
    std::map<std::tuple<int, int, int>, uint32_t> corners;
    for (const tinyobj::shape_t& shape : shapes){
        for (const tinyobj::index_t& index : shape.mesh.indices){
            const int v = index.vertex_index, t = index.texcoord_index, n = index.normal_index;
            if (v < 0 || static_cast<size_t>(v) >= vertexCount) fail("obj vertex index is out of range in " + path);
            const auto [it, added] = corners.try_emplace({v, t, n}, static_cast<uint32_t>(v));
            if (added && claimed[v]){
                it->second = static_cast<uint32_t>(mesh.positions.size());
                const glm::vec3 position = mesh.positions[v];
                mesh.positions.push_back(position);
                if (hasUv) mesh.texcoords.emplace_back();
                if (hasNormal) mesh.normals.emplace_back();
            }
            if (added){
                claimed[v] = 1;
                if (hasUv && t >= 0) mesh.texcoords[it->second] = {attrib.texcoords.at(2 * t), attrib.texcoords.at(2 * t + 1)};
                if (hasNormal && n >= 0){
                    mesh.normals[it->second] = {attrib.normals.at(3 * n), attrib.normals.at(3 * n + 1), attrib.normals.at(3 * n + 2)};
                }
            }
            mesh.indices.push_back(it->second);
        }
    }
    return mesh;
}

std::string str(tg3_str s){
    return s.data ? std::string(s.data, s.len) : std::string();
}

template <class T>
T load(const uint8_t* p){
    T v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Float, or an integer type allowed by KHR_mesh_quantization.
float component(const uint8_t* p, int32_t type, bool normalized){
    switch (type){
        case TG3_COMPONENT_TYPE_FLOAT: return load<float>(p);
        case TG3_COMPONENT_TYPE_BYTE: return normalized ? std::max(load<int8_t>(p) / 127.0f, -1.0f) : load<int8_t>(p);
        case TG3_COMPONENT_TYPE_UNSIGNED_BYTE: return normalized ? *p / 255.0f : *p;
        case TG3_COMPONENT_TYPE_SHORT: return normalized ? std::max(load<int16_t>(p) / 32767.0f, -1.0f) : load<int16_t>(p);
        case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: return normalized ? load<uint16_t>(p) / 65535.0f : load<uint16_t>(p);
        default: fail("glTF attribute has an unsupported component type");
    }
}

// The parser checks buffer views against byteLength, not against the bytes it actually loaded.
const uint8_t* viewBytes(const tg3_model& model, int32_t index){
    const tg3_buffer_view& view = model.buffer_views[index];
    const tg3_span_u8& data = model.buffers[view.buffer].data;
    if (view.byte_offset + view.byte_length > data.count) fail("glTF buffer data is missing");
    return data.data + view.byte_offset;
}

// read(elementBytes) for each element. tg3_parse_file has already range-checked every index and byte span.
template <class Read>
auto readAccessor(const tg3_model& model, int32_t index, int32_t type, Read read){
    const tg3_accessor& accessor = model.accessors[index];
    if (accessor.type != type) fail("glTF accessor has an unexpected type");
    if (accessor.sparse.is_sparse || accessor.buffer_view < 0) fail("sparse or buffer-less glTF accessors are not supported");
    const uint8_t* data = viewBytes(model, accessor.buffer_view) + accessor.byte_offset;
    const size_t stride = tg3_accessor_byte_stride(&accessor, &model.buffer_views[accessor.buffer_view]);
    std::vector<decltype(read(data))> out(accessor.count);
    for (size_t i = 0; i < out.size(); ++i) out[i] = read(data + i * stride);
    return out;
}

// TG3_TYPE_VEC2 and TG3_TYPE_VEC3 equal their component counts.
template <int N>
std::vector<glm::vec<N, float>> readFloats(const tg3_model& model, int32_t index){
    const tg3_accessor& accessor = model.accessors[index];
    const int32_t size = tg3_component_size(accessor.component_type);
    return readAccessor(model, index, N, [&](const uint8_t* p){
        glm::vec<N, float> v;
        for (int c = 0; c < N; ++c) v[c] = component(p + c * size, accessor.component_type, accessor.normalized);
        return v;
    });
}

std::vector<uint32_t> readIndices(const tg3_model& model, int32_t index){
    const int32_t type = model.accessors[index].component_type;
    return readAccessor(model, index, TG3_TYPE_SCALAR, [type](const uint8_t* p) -> uint32_t{
        switch (type){
            case TG3_COMPONENT_TYPE_UNSIGNED_BYTE: return *p;
            case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: return load<uint16_t>(p);
            case TG3_COMPONENT_TYPE_UNSIGNED_INT: return load<uint32_t>(p);
            default: fail("glTF indices must be unsigned integers");
        }
    });
}

std::vector<uint32_t> triangleList(const std::vector<uint32_t>& c, int32_t mode){
    if (mode == TG3_MODE_TRIANGLES){
        if (c.size() % 3 != 0) fail("glTF triangle index count is not a multiple of 3");
        return c;
    }
    std::vector<uint32_t> out;
    for (size_t i = 0; i + 2 < c.size(); ++i){
        if (mode == TG3_MODE_TRIANGLE_FAN) out.insert(out.end(), {c[0], c[i + 1], c[i + 2]});
        else if (i % 2 == 0) out.insert(out.end(), {c[i], c[i + 1], c[i + 2]});
        else out.insert(out.end(), {c[i + 1], c[i], c[i + 2]});
    }
    return out;
}

glm::mat4 nodeMatrix(const tg3_node& node){
    if (node.has_matrix) return glm::mat4(glm::make_mat4(node.matrix));
    // glTF stores quaternions as xyzw. glm::quat takes wxyz.
    const glm::quat rotation(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                             static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]));
    return glm::translate(glm::mat4(1.0f), glm::vec3(glm::make_vec3(node.translation))) * glm::mat4_cast(rotation) *
           glm::scale(glm::mat4(1.0f), glm::vec3(glm::make_vec3(node.scale)));
}

glm::mat3 normalMatrix(const glm::mat4& xform){
    const glm::mat3 linear(xform);
    if (std::abs(glm::determinant(linear)) < 1e-12f) return linear;
    return glm::transpose(glm::inverse(linear));
}

int32_t attribute(const tg3_primitive& prim, const char* name){
    for (uint32_t i = 0; i < prim.attributes_count; ++i){
        if (tg3_str_equals_cstr(prim.attributes[i].key, name)) return prim.attributes[i].value;
    }
    return -1;
}

struct GltfReader{
    const tg3_model& model;
    Mesh mesh;
    bool hasUv = false;
    bool hasNormal = false;

    void primitive(const tg3_primitive& prim, const glm::mat4& xform, const std::string& name){
        if (prim.mode != TG3_MODE_TRIANGLES && prim.mode != TG3_MODE_TRIANGLE_STRIP && prim.mode != TG3_MODE_TRIANGLE_FAN) return;
        const int32_t positionAccessor = attribute(prim, "POSITION");
        const int32_t uvAccessor = attribute(prim, "TEXCOORD_0");
        const int32_t normalAccessor = attribute(prim, "NORMAL");
        if (positionAccessor < 0) fail("glTF primitive is missing POSITION");
        const std::vector<glm::vec3> positions = readFloats<3>(model, positionAccessor);
        const auto uvs = uvAccessor >= 0 ? readFloats<2>(model, uvAccessor) : std::vector<glm::vec2>(positions.size());
        const auto normals = normalAccessor >= 0 ? readFloats<3>(model, normalAccessor) : std::vector<glm::vec3>(positions.size());
        if (uvs.size() != positions.size() || normals.size() != positions.size()){
            fail("glTF attribute count does not match POSITION");
        }

        std::vector<uint32_t> corners(positions.size());
        if (prim.indices >= 0) corners = readIndices(model, prim.indices);
        else std::iota(corners.begin(), corners.end(), 0u);
        corners = triangleList(corners, prim.mode);
        if (corners.empty()) return;

        const auto base = static_cast<uint32_t>(mesh.positions.size());
        mesh.parts.push_back({.name = name, .vertexOffset = base, .vertexCount = static_cast<uint32_t>(positions.size()),
                              .indexOffset = static_cast<uint32_t>(mesh.indices.size()),
                              .indexCount = static_cast<uint32_t>(corners.size()), .material = prim.material});
        for (uint32_t corner : corners){
            if (corner >= positions.size()) fail("glTF index is out of range");
            mesh.indices.push_back(base + corner);
        }
        const glm::mat3 nmat = normalMatrix(xform);
        for (size_t i = 0; i < positions.size(); ++i){
            mesh.positions.push_back(glm::vec3(xform * glm::vec4(positions[i], 1.0f)));
            const glm::vec3 normal = nmat * normals[i];
            const float len = glm::length(normal);
            mesh.normals.push_back(len > 0.0f ? normal / len : glm::vec3(0.0f));
        }
        mesh.texcoords.insert(mesh.texcoords.end(), uvs.begin(), uvs.end());
        hasUv |= uvAccessor >= 0;
        hasNormal |= normalAccessor >= 0;
    }

    void addMesh(const tg3_mesh& gltfMesh, const glm::mat4& xform, const std::string& name){
        for (uint32_t i = 0; i < gltfMesh.primitives_count; ++i) primitive(gltfMesh.primitives[i], xform, name);
    }

    // Skinning and morph targets stay in the node's bind pose.
    void node(int32_t index, const glm::mat4& parent, uint32_t depth){
        if (depth > model.nodes_count) fail("glTF node hierarchy has a cycle");
        const tg3_node& n = model.nodes[index];
        const glm::mat4 world = parent * nodeMatrix(n);
        if (n.mesh >= 0){
            const tg3_mesh& gltfMesh = model.meshes[n.mesh];
            addMesh(gltfMesh, world, n.name.len ? str(n.name) : str(gltfMesh.name));
        }
        for (uint32_t i = 0; i < n.children_count; ++i) node(n.children[i], world, depth + 1);
    }
};

Mesh loadGltf(const std::string& path){
    tg3_parse_options opts;
    tg3_parse_options_init(&opts);
    tinygltf3::Model gltf;
    tinygltf3::ErrorStack errors;
    if (tinygltf3::parse_file(gltf, errors, path.c_str(), &opts) != TG3_OK){
        std::string message = "failed to read " + path;
        for (uint32_t i = 0; i < errors.count(); ++i){
            const tg3_error_entry* entry = errors.entry(i);
            if (entry->severity == TG3_SEVERITY_ERROR && entry->message) message += std::string("; ") + entry->message;
        }
        fail(message);
    }

    const tg3_model& model = *gltf.get();
    GltfReader reader{.model = model, .mesh = {.format = MeshFormat::Gltf}};
    if (model.scenes_count > 0){
        const tg3_scene& scene = model.scenes[std::max(model.default_scene, 0)];
        for (uint32_t i = 0; i < scene.nodes_count; ++i) reader.node(scene.nodes[i], glm::mat4(1.0f), 0);
    } else{
        for (uint32_t i = 0; i < model.meshes_count; ++i) reader.addMesh(model.meshes[i], glm::mat4(1.0f), str(model.meshes[i].name));
    }
    Mesh& mesh = reader.mesh;
    if (!reader.hasUv) mesh.texcoords.clear();
    if (!reader.hasNormal) mesh.normals.clear();

    for (uint32_t i = 0; i < model.images_count; ++i){
        const tg3_image& image = model.images[i];
        MeshImage& out = mesh.images.emplace_back(MeshImage{str(image.name), str(image.mime_type), {}});
        if (image.buffer_view < 0) continue;
        const uint8_t* bytes = viewBytes(model, image.buffer_view);
        out.encoded.assign(bytes, bytes + model.buffer_views[image.buffer_view].byte_length);
    }
    for (uint32_t i = 0; i < model.materials_count; ++i){
        const tg3_material& material = model.materials[i];
        const int32_t texture = material.pbr_metallic_roughness.base_color_texture.index;
        mesh.materials.push_back({str(material.name), texture >= 0 ? model.textures[texture].source : -1});
    }
    return std::move(mesh);
}

float boxVolume(glm::vec3 low, glm::vec3 high){
    const glm::vec3 extent = glm::max(high - low, glm::vec3(0.0f));
    return extent.x * extent.y * extent.z;
}

}  // namespace

Mesh dropOverlappingParts(Mesh mesh){
    if (mesh.parts.size() < 2) return mesh;
    Mesh kept{.format = mesh.format, .images = mesh.images, .materials = mesh.materials};
    std::vector<std::pair<glm::vec3, glm::vec3>> keptBounds;
    for (const MeshPart& part : mesh.parts){
        glm::vec3 low(std::numeric_limits<float>::infinity()), high(-low);
        for (uint32_t i = 0; i < part.vertexCount; ++i){
            low = glm::min(low, mesh.positions[part.vertexOffset + i]);
            high = glm::max(high, mesh.positions[part.vertexOffset + i]);
        }
        const auto duplicates = [&](size_t k){
            const auto [keptLow, keptHigh] = keptBounds[k];
            const float smaller = std::min(boxVolume(low, high), boxVolume(keptLow, keptHigh));
            return kept.parts[k].material == part.material && smaller > 0.0f &&
                   boxVolume(glm::max(low, keptLow), glm::min(high, keptHigh)) > 0.9f * smaller;
        };
        bool duplicate = false;
        for (size_t k = 0; k < kept.parts.size() && !duplicate; ++k) duplicate = duplicates(k);
        if (duplicate) continue;

        MeshPart& copy = kept.parts.emplace_back(part);
        copy.vertexOffset = static_cast<uint32_t>(kept.positions.size());
        copy.indexOffset = static_cast<uint32_t>(kept.indices.size());
        keptBounds.emplace_back(low, high);
        const auto vertices = [&](const auto& from, auto& to){
            if (!from.empty()) to.insert(to.end(), from.begin() + part.vertexOffset, from.begin() + part.vertexOffset + part.vertexCount);
        };
        vertices(mesh.positions, kept.positions);
        vertices(mesh.normals, kept.normals);
        vertices(mesh.texcoords, kept.texcoords);
        for (uint32_t i = 0; i < part.indexCount; ++i){
            kept.indices.push_back(mesh.indices[part.indexOffset + i] - part.vertexOffset + copy.vertexOffset);
        }
    }
    return kept;
}

Mesh loadMesh(const std::string& path){
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    Mesh mesh;
    if (ext == ".obj") mesh = loadObj(path);
    else if (ext == ".gltf" || ext == ".glb") mesh = loadGltf(path);
    else fail("unsupported mesh extension \"" + ext + "\" for " + path);
    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0) fail(path + " has no triangles");
    return mesh;
}
