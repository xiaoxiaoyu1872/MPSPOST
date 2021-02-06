#include <string>
#include <iostream>

#include "GPUPOST.h"


GPUPOST::GPUPOST()
{
    params = new Params();
    gMemory  = new GPUmemory();
}

GPUPOST::~GPUPOST()
{
    delete params;
    delete gMemory;
    delete fileData;
}

void GPUPOST::initialize()
{
    checkMemUsed();

    params->setFilname();
    params->setParams();
    params->printInfo();

    params->setGPUId();
    params->printGPUInfo();

    gMemory->initGPUmemory(params);
}

void GPUPOST::runSimulation()
{
    fileData = new FileData(params,gMemory);

    NeighborSearch neighborSearch(gMemory, params);

    // if (params->mConfigParams.isSurface)
        SurfaceReconstruction surfaceReconstruction(gMemory, params, fileData);

    // if (params->mConfigParams.isDiffuse)
        // DiffuseGeneration diffuseGeneration(gMemory, params, fileData);

    if (params->mConfigParams.isDiffuse)
    {
        fileData->loadBoundFile();
        neighborSearch.BoundGridBuilding();
    }

    for (uint frameIndex = params->mConfigParams.frameStart; 
              frameIndex <= params->mConfigParams.frameEnd; 
              frameIndex += params->mConfigParams.frameStep)
    {
        checkMemUsed();
        gMemory->Memreset();

        fileData->loadFluidFiledat(frameIndex);

        neighborSearch.SpatialGridBuilding();

        if (params->mConfigParams.isSurface)
            surfaceReconstruction.runsimulation();

        // if (params->mConfigParams.isDiffuse)
        //     diffuseGeneration.runsimulation();

        gMemory->Memfree();
    }
} 


void GPUPOST::finalize()
{
    gMemory->finaGPUmemory();
}