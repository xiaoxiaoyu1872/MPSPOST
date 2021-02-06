#ifndef SURFACERECONSTRUCTION_H_
#define SURFACERECONSTRUCTION_H_

#include "Params.h"
#include "GPUmemory.h"
#include "myCuda.cuh"
#include "FileData.h"

#include "MethodOfYu.h"
#include "MarchingCube.h"

class SurfaceReconstruction
{
public:
    SurfaceReconstruction(GPUmemory *gMemory, Params *params, FileData *fileData);
    ~SurfaceReconstruction();
    void runsimulation();
private:
    GPUmemory* gMemory;
    Params* params;
    FileData* fileData;

    MethodOfYu* methodOfYu;
    MarchingCube* marchingCube;

    void saveMiddleFile();

};

#endif 