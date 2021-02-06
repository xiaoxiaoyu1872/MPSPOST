#include "MethodOfYu.h"
#include "MethodOfYu_kernel.cu"
#include "Thrust.cuh"

MethodOfYu::MethodOfYu(GPUmemory *_gMemory, Params *_params)
{
    params = _params;
    gMemory = _gMemory;

	gMemory->SurfaceAlloFixedMem();
    constantMemCopy();

    int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

	cudaStreamCreateWithPriority(&s1, cudaStreamDefault, priority_high);
	cudaStreamCreateWithPriority(&s2, cudaStreamDefault, priority_low);
}

MethodOfYu::~MethodOfYu()
{
    cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);

	checkMemUsed();

	gMemory->SurfaceFreeFixedMem();
	std::cout << "~~MethodOfYu" << std::endl;

	checkMemUsed();
}


void MethodOfYu::processingOfParticles()
{
    extractionOfSurfaceAndInvolveParticles();
    thrustscan_particles();
    memallocation_particles();
    streamcompact_particles();
    smoothedparticles();
    transformmatrices();
}


void MethodOfYu::processingOfVertices()
{
	extractionOfSurfaceVertices();
    thrustscan_vertices();
	memallocation_vertices();
    streamcompact_vertices();
}


void MethodOfYu::estimationOfscalarField()
{
	scalarfield();
}


void MethodOfYu::extractionOfSurfaceAndInvolveParticles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(params->mGridParams.spSize, gridDim, blockDim); 

    estimationOfSurfaceParticles <<< gridDim, blockDim >>> (
		gMemory->dSpatialGrid,
		gMemory->dNumSurParticleGrid,
		gMemory->dNumInvParticleGrid,
		gMemory->dIndexRange);

	cudaDeviceSynchronize();

	// estimationOfInvolveParticles  <<< gridDim, blockDim >>> (
	// 	gMemory->dNumSurParticleGrid,
	// 	gMemory->dNumInvParticleGrid);

	// cudaDeviceSynchronize();
}

void MethodOfYu::thrustscan_particles()
{
    NumSurfaceParticles = ThrustExclusiveScan(
		gMemory->dNumSurParticleGridScan,
		gMemory->dNumSurParticleGrid,
		(uint)params->mGridParams.spSize);

	gMemory->NumSurfaceParticles = NumSurfaceParticles;

	if (NumSurfaceParticles == 0)
	{
		std::cerr << "No surface particle detected!\n";
		return;
	}

	std::cout << "mNumSurfaceParticles =  " << NumSurfaceParticles << std::endl;

	std::cout << "surface particles ratio: " << 
	static_cast<double>(NumSurfaceParticles)
		/ gMemory->NumParticles << std::endl;


	// NumInvolveParticles = ThrustExclusiveScan(
	// 	gMemory->dNumInvParticleGridScan,
	// 	gMemory->dNumInvParticleGrid,
	// 	(uint)params->mGridParams.spSize);

	// gMemory->NumInvolveParticles = NumInvolveParticles;


	// if (NumInvolveParticles == 0)
	// {
	// 	std::cerr << "No involve particle detected!\n";
	// 	return;
	// }

	// std::cout << "mNumInvolveParticles =  " << NumInvolveParticles << std::endl;

	// std::cout << "involve particles ratio: " << 
	// static_cast<double>(NumInvolveParticles)
	// 	/ gMemory->NumParticles << std::endl;
}


void MethodOfYu::streamcompact_particles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(params->mGridParams.spSize, gridDim, blockDim);

	compactationOfParticles << < gridDim, blockDim, 0, s1>> > (
		gMemory->dNumSurParticleGrid,
		gMemory->dNumSurParticleGridScan,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex);	
	cudaStreamSynchronize(s1);

	// compactationOfParticles << < gridDim, blockDim, 0,  s2>> > (
	// 	gMemory->dNumInvParticleGrid,
	// 	gMemory->dNumInvParticleGridScan,
	// 	gMemory->dIndexRange,
	// 	gMemory->dInvolveParticlesIndex);
	// cudaStreamSynchronize(s2);
}

void MethodOfYu::smoothedparticles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceParticles, gridDim, blockDim);
	calculationOfSmoothed << < gridDim, blockDim , 0, s1>> > (
		gMemory->dFluidParticle,
		gMemory->dSurfaceParticlesMean,
		gMemory->dSurfaceParticlesSmoothed,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex,
		gMemory->dSpatialGrid);
	cudaStreamSynchronize(s1);
	
	// calcGridDimBlockDim(NumInvolveParticles, gridDim, blockDim);
	// calculationOfSmoothedforInvovle << < gridDim, blockDim , 0, s2>> > (
	// 	gMemory->dFluidParticle,
	// 	gMemory->dInvolveParticlesSmoothed,
	// 	gMemory->dIndexRange,
	// 	gMemory->dInvolveParticlesIndex);
	// cudaStreamSynchronize(s2);
}


void MethodOfYu::transformmatrices()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceParticles, gridDim, blockDim);
	calculationOfTransformMatrices << < gridDim, blockDim, 0, s1 >> > (
		gMemory->dSurfaceParticlesMean,
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex,
		gMemory->dSVDMatrices);

	cudaStreamSynchronize(s1);
}


void MethodOfYu::extractionOfSurfaceVertices()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceParticles, gridDim, blockDim);
	estimationOfSurfaceVertices << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dSurfaceParticlesSmoothed,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex,
		gMemory->dIsSurfaceVertices,
		gMemory->dSVDMatrices,
		gMemory->dSpatialGrid);

	cudaDeviceSynchronize();
}

void MethodOfYu::thrustscan_vertices()
{
	NumSurfaceVertices = ThrustExclusiveScan(
		gMemory->dIsSurfaceVerticesScan,
		gMemory->dIsSurfaceVertices,
		(uint)params->mGridParams.scSize);

	gMemory->NumSurfaceVertices = NumSurfaceVertices;

	if (NumSurfaceVertices == 0)
	{
		std::cerr << "No surface vertex detected!\n";
		return;
	}

	std::cout << "mNumSurfaceVertices =  " << NumSurfaceVertices << std::endl;

	std::cout << "surface vertices ratio: " << static_cast<double>(NumSurfaceVertices) /
		(params->mGridParams.scSize) << std::endl;
}

void MethodOfYu::streamcompact_vertices()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(params->mGridParams.scSize, gridDim, blockDim);

	compactationOfSurfaceVertices << < gridDim, blockDim>> > (
		gMemory->dIsSurfaceVertices,
		gMemory->dIsSurfaceVerticesScan,
		gMemory->dSurfaceVerticesIndex);

	cudaDeviceSynchronize();
}

void MethodOfYu::scalarfield()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceVertices, gridDim, blockDim);

	computationOfScalarFieldGrid << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dSurfaceParticlesSmoothed,
		gMemory->dInvolveParticlesSmoothed,
		gMemory->dIndexRange,
		gMemory->dSurfaceParticlesIndex,
		gMemory->dInvolveParticlesIndex,
		gMemory->dSurfaceVerticesIndex,
		gMemory->dNumSurParticleGrid,
		gMemory->dNumSurParticleGridScan,
		gMemory->dNumInvParticleGrid,
		gMemory->dNumInvParticleGridScan,
		gMemory->dSVDMatrices,
		gMemory->dScalarFiled);

	cudaDeviceSynchronize();
}


void MethodOfYu::constantMemCopy()
{
    checkCudaErrors(cudaMemcpyToSymbol(dSurfaceParams, &params->mSurfaceParams, sizeof(SurfaceParams)));

	checkCudaErrors(cudaMemcpyToSymbol(dSimParams, &params->mSimParams, sizeof(SimParams)));

	checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
}


void MethodOfYu::constantMemSurAndInvPar_Num()
{
	checkCudaErrors(cudaMemcpyToSymbol(dNumSurfaceParticles, 
	&gMemory->NumSurfaceParticles, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(dNumInvolveParticles, 
	&gMemory->NumInvolveParticles, sizeof(uint)));	
}

void MethodOfYu::constantMemSurVer_Num()
{
	checkCudaErrors(cudaMemcpyToSymbol(dNumSurfaceVertices, 
	&gMemory->NumSurfaceVertices, sizeof(uint)));
}

void MethodOfYu::memallocation_particles()
{
    gMemory->memAllocation_particles();
	constantMemSurAndInvPar_Num();
}

void MethodOfYu::memallocation_vertices()
{
	gMemory->memAllocation_vertices();
	constantMemSurVer_Num();
}

