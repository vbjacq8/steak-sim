#include"open3d/Open3D.h"
#include<glm/glm.hpp>
#include<vector>

namespace o3c = open3d::core;


class VoxelFactory {
    

    public:
    
    // Returns nx*ny*nz values, 1 = inside the mesh. Index = x + nx*(y + ny*z).
    static std::vector<float> voxelizeSolid(const std::vector<glm::vec3>& positions,
                                    const std::vector<uint32_t>& indices,
                                    glm::vec3 origin, float voxelSize,
                                    int nx, int ny, int nz);
    };