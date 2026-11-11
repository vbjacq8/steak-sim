// Compute-shader voxelizer.
//
// Each thread owns one voxel and tests it against every triangle with an exact
// triangle/box overlap test (surface), and counts +Y ray crossings for a parity
// inside test (interior). Whole workgroups skip triangles that cannot touch their block.
//
// When captureUv is set, occupied voxels also store the barycentric UV of the closest
// point on the mesh. Interior voxels have no triangle of their own, so they inherit the
// nearest surface point's UV: the crust texture bleeds inward. Empty voxels store (-1, -1).

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
    uint captureUv; // 1: write closest-point UVs for occupied voxels
};

[[vk::push_constant]] ConstantBuffer<VoxelizeParams> params;

[[vk::binding(0,0)]] StructuredBuffer<float> vertexData;
[[vk::binding(1,0)]] StructuredBuffer<uint> indexData;
[[vk::binding(2,0)]] RWStructuredBuffer<uint> voxels; // index = x + nx * (y + ny * z)
[[vk::binding(3,0)]] StructuredBuffer<float2> vertexUv; // one UV per vertex, indexed like positions
[[vk::binding(4,0)]] RWStructuredBuffer<float2> voxelUv;

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

float2 loadUv(uint index){
    return vertexUv[indexData[index]];
}

// Barycentric weights (wa, wb, wc) of the closest point on triangle abc to p.
float3 closestBarycentric(float3 a, float3 b, float3 c, float3 p){
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return float3(1.0, 0.0, 0.0);

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return float3(0.0, 1.0, 0.0);

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0){
        float v = d1 / (d1 - d3);
        return float3(1.0 - v, v, 0.0);
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return float3(0.0, 0.0, 1.0);

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0){
        float w = d2 / (d2 - d6);
        return float3(1.0 - w, 0.0, w);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0){
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return float3(0.0, 1.0 - w, w);
    }

    float denom = va + vb + vc;
    if (abs(denom) < 1e-20) return float3(1.0, 0.0, 0.0);
    float v = vb / denom;
    float w = vc / denom;
    return float3(1.0 - v - w, v, w);
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
    uint voxelIndex = voxel.x + params.dims.x * (voxel.y + params.dims.y * voxel.z);
    voxels[voxelIndex] = flags;

    // Second pass only for occupied voxels, so empty space does not pay for the UV query.
    // Interior voxels get the nearest surface point because the mesh has no interior UVs.
    if (params.captureUv != 0){
        if (flags == 0){
            voxelUv[voxelIndex] = float2(-1.0, -1.0);
        } else {
            float bestDist2 = 1e30;
            float2 bestUv = float2(0.0, 0.0);
            for (uint t = 0; t < params.triangleCount; ++t){
                float3 a = loadPosition(3 * t);
                float3 b = loadPosition(3 * t + 1);
                float3 c = loadPosition(3 * t + 2);
                float3 triMin = min(a, min(b, c));
                float3 triMax = max(a, max(b, c));
                float3 clamped = clamp(center, triMin, triMax);
                float bound = dot(center - clamped, center - clamped);
                if (bound >= bestDist2) continue;

                float3 bary = closestBarycentric(a, b, c, center);
                float3 closest = bary.x * a + bary.y * b + bary.z * c;
                float dist2 = dot(closest - center, closest - center);
                if (dist2 < bestDist2){
                    bestDist2 = dist2;
                    bestUv = bary.x * loadUv(3 * t) + bary.y * loadUv(3 * t + 1) + bary.z * loadUv(3 * t + 2);
                }
            }
            voxelUv[voxelIndex] = bestUv;
        }
    }
}
