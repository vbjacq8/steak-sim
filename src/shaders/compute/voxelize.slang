// Compute-shader voxelizer.
//
// Each thread owns one voxel and tests it against every triangle with an exact
// triangle/box overlap test (surface), and counts +Y ray crossings for a parity
// inside test (interior). Whole workgroups skip triangles that cannot touch their block.

import voxel_math;

static const int GROUP_SIZE = 4; //Workgroup size

static const int VOXEL_INSIDE = 1u; //Constants used to calculate bit flags for each voxel
static const int VOXEL_OUTSIDE = 2u;

struct VoxelizeParams {
    float3 origin;
    float voxelSize; // width of a voxel
    uint3 dims;
    uint triangleCount; // number of triangles to voxelize
    uint vertexStride; // how many floats exist per vertex (in our case it is 3 for position)
};

[[vk::push_constant]] ConstantBuffer<VoxelizeParams> params;

[[vk::binding(0,0)]] StructuredBuffer<float> vertexData;
[[vk::binding(1,0)]] StructuredBuffer<uint> indexData;
[[vk::binding(2,0)]] RWStructuredBuffer<uint> voxels; // index = x + nx * (y + ny * z)

/**
 * \brief returns a float3 representing the position
 * \param index takes index calculated off raw triangle data; ex: index = 3 corresponds to the vertex index of 2nd triangle
 */
float3 loadPosition(uint index){
    // parameter is in vertex representation; want the index for the first float of this vertex (x position)
    uint base = indexData[index] * params.vertexStride;
    // put back together our floats to return the actual position
    return float3(vertexData[base], vertexData[base+1], vertexData[base+2]);
}

[shader("compute")]
[numthreads(GROUP_SIZE, GROUP_SIZE, GROUP_SIZE)]
void voxelizeMain(uint3 voxel : SV_DispatchThreadID, uint3 group : SV_GroupID) {
    if (any(voxel >= params.dims)) {
        return;
    }
    float s = params.voxelSize;
    float3 halfSize = 0.5 * s;
    float3 center = params.origin + (float3(voxel) + 0.5) * s;
     // Off-center so the ray never passes exactly through a shared edge or vertex,
    // which would count one surface crossing twice and flip the parity.
    float3 rayOrigin = center + float3(1.69e-4, 0.0, 2.67e-4) * s;

    // calculate the bounds for a workgroup / block so that we can reject any triangles not in it
    float3 blockMin = params.origin + float3(group * GROUP_SIZE) * s;
    float3 blockMax = blockMin + GROUP_SIZE * s;

    bool surface = false;
    uint crossings = 0;
    for (uint t = 0; t < params.triangleCount; ++t){
        //load the three vertices of each triangle
        float3 a = loadPosition(3 * t);
        float3 b = loadPosition(3 * t + 1);
        float3 c = loadPosition(3 * t + 2);
        float3 triMin = min(a, min(b,c));
        float3 triMax = max(a, max(b,c));

        if (!surface && all(triMin <= blockMax) && all(triMax >= blockMin)){
            surface = triangleBoxOverlap(a,b,c, center, halfSize);

        }

        bool overFootprint = triMin.x <= blockMax.x && triMax.x >= blockMin.x &&
            triMin.z <= blockMax.z && triMax.z >= blockMin.z;
        if (overFootprint && triMax.y > blockMin.y && rayCrossesUp(a,b,c,rayOrigin)){
            ++crossings;
        }
    }
    //create flag corresponding to the values; assign the voxel map that value.
    uint flags = ((crossings & 1u) != 0u ? VOXEL_INSIDE : 0u) | (surface ? VOXEL_OUTSIDE : 0u);
    voxels[voxel.x + params.dims.x * (voxel.y + params.dims.y * voxel.z)] = flags;


}
