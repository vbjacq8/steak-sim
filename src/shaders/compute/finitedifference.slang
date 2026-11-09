// Compute-shader finite-difference step of the heat equation dT/dt = alpha * laplacian(T).
//
// Forward Euler in time, second-order 7-point Laplacian in space:
//   T'[c] = T[c] + r * sum over the 6 neighbors n of w_n * (T[n] - T[c]),  r = alpha dt / dx^2
// Each thread owns one cell and reads its neighbors straight from global memory. x is the fastest
// index and the workgroup is 32 wide in x, so neighboring threads load neighboring floats.
//
// Every cell has a kind (see heat_solver.hpp), which sets the neighbor weight w_n:
//   air    held at airTemp; heat crosses the air/meat face at h (T_air - T).  w = airCoupling
//   solid  steps the heat equation.                                            w = 1
//   pan    held at panTemp; heat crosses the pan/meat face at h (T_pan - T).  w = panCoupling
// A neighbor outside the grid is the cell itself, so those faces are insulated too.

static const uint GROUP_X = 32;  // must match GROUP_SIZE in heat_solver.cpp
static const uint GROUP_Y = 4;
static const uint GROUP_Z = 2;

static const uint CELL_AIR = 0u;  // must match CELL_* in heat_solver.hpp
static const uint CELL_SOLID = 1u;
static const uint CELL_PAN = 2u;

// Must match HeatStepParams in heat_solver.cpp.
struct HeatStepParams {
    uint3 dims;         // nx, ny, nz
    float fourier;      // r = alpha * dt / dx^2
    float panTemp;      // temperature of pan cells
    float panCoupling;  // pan face conductance / cell face conductance, see HeatParams::panCoupling
    float airTemp;      // temperature of air cells
    float airCoupling;  // air face conductance / cell face conductance, see HeatParams::airCoupling
};

[[vk::push_constant]] ConstantBuffer<HeatStepParams> params;

[[vk::binding(0, 0)]] StructuredBuffer<float> tempIn;     // index = x + nx * (y + ny * z)
[[vk::binding(1, 0)]] RWStructuredBuffer<float> tempOut;
[[vk::binding(2, 0)]] StructuredBuffer<uint> cellKind;

// w_n * (T[n] - T[c]) for neighbor j; j == i for a neighbor outside the grid.
float faceFlux(uint j, float c) {
    uint kind = cellKind[j];
    float difference = tempIn[j] - c;
    if (kind == CELL_AIR) return params.airCoupling * difference;
    if (kind == CELL_PAN) return params.panCoupling * difference;
    return difference;
}

[shader("compute")]
[numthreads(GROUP_X, GROUP_Y, GROUP_Z)]
void heatStepMain(uint3 cell : SV_DispatchThreadID) {
    uint3 n = params.dims;
    if (any(cell >= n)) {
        return;
    }
    uint i = cell.x + n.x * (cell.y + n.y * cell.z);
    uint kind = cellKind[i];
    if (kind == CELL_PAN) {
        tempOut[i] = params.panTemp;
        return;
    }
    float c = tempIn[i];
    if (kind == CELL_AIR) {
        tempOut[i] = params.airTemp;
        return;
    }

    // Pick indices first so no load ever goes out of bounds.
    uint strideY = n.x;
    uint strideZ = n.x * n.y;
    uint xm = cell.x > 0 ? i - 1 : i;
    uint xp = cell.x + 1 < n.x ? i + 1 : i;
    uint ym = cell.y > 0 ? i - strideY : i;
    uint yp = cell.y + 1 < n.y ? i + strideY : i;
    uint zm = cell.z > 0 ? i - strideZ : i;
    uint zp = cell.z + 1 < n.z ? i + strideZ : i;

    float flux = (faceFlux(xm, c) + faceFlux(xp, c)) + (faceFlux(ym, c) + faceFlux(yp, c)) +
                 (faceFlux(zm, c) + faceFlux(zp, c));
    tempOut[i] = c + params.fourier * flux;
}
