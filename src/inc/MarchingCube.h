#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "Params.h"
#include "GPUmemory.h"

#include "myCuda.cuh"


class MarchingCube
{
public:
    MarchingCube(GPUmemory *gMemory, Params *params);
    ~MarchingCube();
    void triangulation();
private:
    GPUmemory* gMemory;
    Params* params;

    uint NumSurfaceVertices;
    uint NumValidSurfaceCubes;
    uint NumSurfaceMeshVertices;

    void memallocation_cubes();
    void detectionOfValidCubes();
    void thrustscan_cubes();
    void streamcompact_cubes();
    void memallocation_triangles();
    void marchingcubes();

    void memallocation_scalarvalue();
    void scalarvalue();

    void constantMemCopy();
    void BindingTexMem();
    void constantMemSurVer_Num();
    void constantMemValCube_Num();
};



#endif