#include"voxelization.hpp"

std::vector<float> VoxelFactory::voxelizeSolid(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    glm::vec3 origin, float voxelSize,
    int nx, int ny, int nz
    ){
        std::vector<float> flat;
        flat.reserve(positions.size() * 3);
        glm::vec3 low(std::numeric_limits<float>::max());
        glm::vec3 high(std::numeric_limits<float>::lowest());
        for (const auto& p : positions) {
            flat.insert(flat.end(), {p.x, p.y, p.z});
            low = glm::min(low, p);
            high = glm::max(high, p);
        }
        const glm::ivec3 lowIdx = glm::ivec3(glm::floor(low/voxelSize)) - glm::ivec3(-1, -1, -1);
        const glm::ivec3 highIdx = glm::ivec3(glm::ceil(hi/voxelSize)) + glm::ivec3(1,1,1);
        const glm::vec3 origin = glm::vec3(loIdx) * voxelSize;

        const int nx = highIdx.x - lowIdx.x;
        const int ny = highIdx.y - lowIdx.y;
        const int nz = highIdx.z - lowIdx.z;

        std::vector<float> centers;
        centers.reserve((size_t)nx * ny * nz * 3);

        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                    centers.insert(centers.end(),
                                {origin.x + (x + 0.5f) * voxelSize,
                                    origin.y + (y + 0.5f) * voxelSize,
                                    origin.z + (z + 0.5f) * voxelSize});
    
        o3c::Tensor query(centers, {(int64_t)nx * ny * nz, 3}, o3c::Float32);
    
        open3d::t::geometry::RaycastingScene scene;
        scene.AddTriangles(
            o3c::Tensor(flat, {(int64_t)positions.size(), 3}, o3c::Float32),
            o3c::Tensor(indices, {(int64_t)indices.size() / 3, 3}, o3c::UInt32));
    
        
        o3c::Tensor occ = scene.ComputeOccupancy(query, 0, 3);
        return occ.To(o3c::Float32).ToFlatVector<float>();


    }