// Compute-shader finite difference temperature derivative solver
//
// Each thread owns one voxel and uses a finite difference stencil scheme to approximate the derivative
// of temperature

import voxel_math;

static const uint GROUP_SIZE = 4;

static const uint VOXEL_INSIDE = 1u;   // voxel center is inside the closed mesh
static const uint VOXEL_SURFACE = 2u;  // voxel box touches at least one triangle

// Must match VoxelizeParams in voxelizer.cpp.
struct VoxelizeParams {
    float3 origin;       // world position of the grid corner
    float voxelSize;
    uint3 dims;          // nx, ny, nz
    uint triangleCount;
    uint vertexStride;   // floats per vertex; position is the first three
};
[[vk::push_constant]] ConstantBuffer<VoxelizeParams> params;

[[vk::binding(0, 0)]] StructuredBuffer<float> vertexData;
[[vk::binding(1, 0)]] StructuredBuffer<uint> indexData;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> voxels;  // index = x + nx * (y + ny * z)

float3 loadPosition(uint index) {
    uint base = indexData[index] * params.vertexStride;
    return float3(vertexData[base], vertexData[base + 1], vertexData[base + 2]);
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
    float3 rayOrigin = center + float3(1.37e-4, 0.0, 2.91e-4) * s;

    // Bounds of this workgroup's block. They are uniform across the group, so the
    // culling branches below never diverge.
    float3 blockMin = params.origin + float3(group * GROUP_SIZE) * s;
    float3 blockMax = blockMin + GROUP_SIZE * s;

    bool surface = false;
    uint crossings = 0;
    for (uint t = 0; t < params.triangleCount; ++t) {
        float3 a = loadPosition(3 * t);
        float3 b = loadPosition(3 * t + 1);
        float3 c = loadPosition(3 * t + 2);
        float3 triMin = min(a, min(b, c));
        float3 triMax = max(a, max(b, c));

        if (!surface && all(triMin <= blockMax) && all(triMax >= blockMin)) {
            surface = triangleBoxOverlap(a, b, c, center, halfSize);
        }

        bool overFootprint = triMin.x <= blockMax.x && triMax.x >= blockMin.x &&
                             triMin.z <= blockMax.z && triMax.z >= blockMin.z;
        if (overFootprint && triMax.y > blockMin.y && rayCrossesUp(a, b, c, rayOrigin)) {
            ++crossings;
        }
    }

    uint flags = ((crossings & 1u) != 0u ? VOXEL_INSIDE : 0u) | (surface ? VOXEL_SURFACE : 0u);
    voxels[voxel.x + params.dims.x * (voxel.y + params.dims.y * voxel.z)] = flags;
}
