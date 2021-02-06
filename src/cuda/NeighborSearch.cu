#include "NeighborSearch.h"
#include "NeighborSearch_kernel.cu"
#include "Thrust.cuh"

NeighborSearch::NeighborSearch(GPUmemory *_gMemory, Params *_params)
{
    gMemory = _gMemory;
    params = _params;
    constantMemCopy_Grid();
}

void NeighborSearch::constantMemCopy_Grid()
{
	checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
}

void NeighborSearch::BoundGridBuilding()
{

    uint NumBoundParticles = gMemory->NumBoundParticles;

    dim3 gridDim, blockDim;
    calcGridDimBlockDim(NumBoundParticles,gridDim, blockDim);

    boundGridBuilding<<< gridDim, blockDim, 0, 0>>>
    (gMemory->dBoundParticle, 
    NumBoundParticles, 
    gMemory->dBoundGrid,
    gMemory->dSpatialGrid);
	getLastCudaError("boundGridBuilding");

    // gMemory->Memfree_bound();
    cudaDeviceSynchronize();
}  


void NeighborSearch::SpatialGridBuilding()
{
    uint NumParticles = gMemory->NumParticles;
    uint *dParticleHash;
	checkCudaErrors(cudaMalloc((void**)&dParticleHash, NumParticles * sizeof(uint)));
   
    dim3 gridDim, blockDim;
    calcGridDimBlockDim(NumParticles,gridDim, blockDim);

    spatialGridBuilding<<< gridDim, blockDim, 0, 0>>>
    (dParticleHash, 
    gMemory->dFluidParticle, 
    NumParticles, 
    gMemory->dSpatialGrid);
	getLastCudaError("spatialGridBuilding");

    cudaDeviceSynchronize();

    ThrustSort(gMemory->dFluidParticle, dParticleHash, NumParticles);

    uint memSize = sizeof(uint) * (numThreads + 1);

    findCellRangeKernel <<< gridDim, blockDim, memSize , 0>>> 
    (gMemory->dIndexRange, 
    NumParticles, 
    dParticleHash, 
    gMemory->dFluidParticle);
	getLastCudaError("findCellRangeKernel");

    safeCudaFree((void**)&dParticleHash);
}  