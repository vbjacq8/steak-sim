// Geometry tests shared by voxelize.slang and the GPU unit tests (tests/voxel_tests.slang).

bool axisSeparates(float3 axis, float3 v0, float3 v1, float3 v2, float3 halfSize) {
    float p0 = dot(v0, axis);
    float p1 = dot(v1, axis);
    float p2 = dot(v2, axis);
    float r = dot(halfSize, abs(axis));
    return min(p0, min(p1, p2)) > r || max(p0, max(p1, p2)) < -r;
}

// Akenine-Möller triangle/box overlap via the separating axis theorem:
// 9 edge-cross axes, the 3 box face normals, then the triangle normal.
// Touching counts as overlapping.
bool triangleBoxOverlap(float3 a, float3 b, float3 c, float3 center, float3 halfSize) {
    float3 v0 = a - center;
    float3 v1 = b - center;
    float3 v2 = c - center;

    float3 edges[3] = { v1 - v0, v2 - v1, v0 - v2 };
    float3 basis[3] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) };
    for (uint e = 0; e < 3; ++e) {
        for (uint i = 0; i < 3; ++i) {
            if (axisSeparates(cross(basis[i], edges[e]), v0, v1, v2, halfSize)) {
                return false;
            }
        }
    }

    if (any(min(v0, min(v1, v2)) > halfSize) || any(max(v0, max(v1, v2)) < -halfSize)) {
        return false;
    }

    return !axisSeparates(cross(edges[0], edges[1]), v0, v1, v2, halfSize);
}

// True if the ray from `origin` toward +Y crosses triangle abc.
bool rayCrossesUp(float3 a, float3 b, float3 c, float3 origin) {
    float2 p = origin.xz;
    float2 pa = a.xz;
    float2 pb = b.xz;
    float2 pc = c.xz;

    float area = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
    if (abs(area) < 1e-10) {
        return false;  // triangle is edge-on to the ray
    }

    float wa = ((pb.x - p.x) * (pc.y - p.y) - (pb.y - p.y) * (pc.x - p.x)) / area;
    float wb = ((pc.x - p.x) * (pa.y - p.y) - (pc.y - p.y) * (pa.x - p.x)) / area;
    float wc = 1.0 - wa - wb;
    if (wa < 0.0 || wb < 0.0 || wc < 0.0) {
        return false;
    }

    return wa * a.y + wb * b.y + wc * c.y > origin.y;
}
