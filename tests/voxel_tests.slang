// Evaluates the voxel_math.slang tests for a batch of triangles so C++ tests can check them.

import voxel_math;

static const uint MODE_TRIANGLE_BOX = 0u;
static const uint MODE_RAY_CROSSES_UP = 1u;

// Must match KernelParams in voxel_tests.cpp.
struct KernelParams {
    float3 point;          // box center (MODE_TRIANGLE_BOX) or ray origin (MODE_RAY_CROSSES_UP)
    uint triangleCount;
    float3 boxHalfSize;
    uint mode;
};
[[vk::push_constant]] ConstantBuffer<KernelParams> params;

[[vk::binding(0, 0)]] StructuredBuffer<float> triangles;  // 9 floats per triangle
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> results;   // 1 = test passed

float3 loadVertex(uint triangle, uint corner) {
    uint base = 9 * triangle + 3 * corner;
    return float3(triangles[base], triangles[base + 1], triangles[base + 2]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void testMain(uint3 id : SV_DispatchThreadID) {
    uint t = id.x;
    if (t >= params.triangleCount) {
        return;
    }

    float3 a = loadVertex(t, 0);
    float3 b = loadVertex(t, 1);
    float3 c = loadVertex(t, 2);
    bool hit = params.mode == MODE_TRIANGLE_BOX
        ? triangleBoxOverlap(a, b, c, params.point, params.boxHalfSize)
        : rayCrossesUp(a, b, c, params.point);
    results[t] = hit ? 1u : 0u;
}
