#include "MarchingCube.h"
#include "MarchingCube_kernel.cu"
#include "Thrust.cuh"

MarchingCube::MarchingCube(GPUmemory *_gMemory, Params *_params)
{
    params = _params;
    gMemory = _gMemory;

    gMemory->AlloTextureMem();
    BindingTexMem();
    constantMemCopy();
}


MarchingCube::~MarchingCube()
{
    checkMemUsed();

    gMemory->FreeTextureMem();
    std::cout << "~~MarchingCube" << std::endl;

    checkMemUsed();
}


void MarchingCube::constantMemCopy()
{
    checkCudaErrors(cudaMemcpyToSymbol(dSurfaceParams, &params->mSurfaceParams, sizeof(SurfaceParams)));
	checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
}


void MarchingCube::triangulation()
{
    memallocation_cubes();

	detectionOfValidCubes();

	thrustscan_cubes();

	memallocation_triangles();

	streamcompact_cubes();

	marchingcubes();

	memallocation_scalarvalue();

	scalarvalue();
}


void MarchingCube::detectionOfValidCubes()
{
	NumSurfaceVertices = gMemory->NumSurfaceVertices;
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceVertices, gridDim, blockDim);
	detectValidSurfaceCubes << <gridDim, blockDim >> > (
		gMemory->dSurfaceVerticesIndex,
		gMemory->dScalarFiled,
		gMemory->dIsSurfaceVertices,
		gMemory->dCubeFlag,
		gMemory->dIsValidSurfaceCube,
		gMemory->dNumVertexCube);
	cudaDeviceSynchronize();
}

void MarchingCube::thrustscan_cubes()
{
	NumValidSurfaceCubes = ThrustExclusiveScan(
		gMemory->dIsValidSurfaceCubeScan,
		gMemory->dIsValidSurfaceCube,
		(uint)NumSurfaceVertices);

	NumSurfaceMeshVertices = ThrustExclusiveScan(
		gMemory->dNumVertexCubeScan,
		gMemory->dNumVertexCube,
		(uint)NumSurfaceVertices);

	gMemory->NumValidSurfaceCubes = NumValidSurfaceCubes;
	gMemory->NumSurfaceMeshVertices = NumSurfaceMeshVertices;

	if (NumValidSurfaceCubes <= 0)
	{
		std::cerr << "No vertex of surface mesh detected!\n";
		return;
	}

	if (NumSurfaceMeshVertices <= 0)
	{
		std::cerr << "No vertex of surface mesh detected!\n";
		return;
	}

	std::cout << "valid surface cubes ratio: " << static_cast<double>(NumValidSurfaceCubes)
	/ (params->mGridParams.scSize) << std::endl;

	std::cout << "surface vertex number: " << static_cast<double>(NumSurfaceMeshVertices) << std::endl;

}

void MarchingCube::streamcompact_cubes()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceVertices, gridDim, blockDim);

	compactValidSurfaceCubes << <gridDim, blockDim >> > (
		gMemory->dValidCubesIndex,
		gMemory->dIsValidSurfaceCube,
		gMemory->dIsValidSurfaceCubeScan);
	cudaDeviceSynchronize();
}

void MarchingCube::marchingcubes()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumValidSurfaceCubes, gridDim, blockDim);

	generateTriangles<< <gridDim, blockDim >> >(
		gMemory->dSurfaceVerticesIndex,
		gMemory->dValidCubesIndex,
		gMemory->dCubeFlag,
		gMemory->dNumVertexCubeScan,
		gMemory->dScalarFiled,
		gMemory->dVertex,
		gMemory->dNormal);
	cudaDeviceSynchronize();
}

void MarchingCube::scalarvalue()
{
	dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumSurfaceMeshVertices, gridDim, blockDim);

	estimationOfScalarValue<< <gridDim, blockDim >> >(
		gMemory->dVertex,
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dScalarValue);
	cudaDeviceSynchronize();
}

void MarchingCube::memallocation_cubes()
{
	constantMemSurVer_Num();
	gMemory->memAllocation_cubes();
}


void MarchingCube::memallocation_triangles()
{
	gMemory->memAllocation_triangles();
	constantMemValCube_Num();
}


void MarchingCube::memallocation_scalarvalue()
{
	gMemory->memAllocation_scalarvalues();
}

void MarchingCube::constantMemSurVer_Num()
{
	checkCudaErrors(cudaMemcpyToSymbol(dNumSurfaceVertices, 
	&gMemory->NumSurfaceVertices, sizeof(uint)));
}

void MarchingCube::constantMemValCube_Num()
{
	checkCudaErrors(cudaMemcpyToSymbol(dNumValidSurfaceCubes, 
	&gMemory->NumValidSurfaceCubes, sizeof(uint)));
	checkCudaErrors(cudaMemcpyToSymbol(dNumSurfaceMeshVertices, 
	&gMemory->NumSurfaceMeshVertices, sizeof(uint)));
}

void MarchingCube::BindingTexMem()
{
	cudaChannelFormatDesc channelDescUnsigned =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaChannelFormatDesc channelDescSigned =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);

	checkCudaErrors(cudaBindTexture(0, edgeTex, gMemory->dEdgeTable, channelDescUnsigned));
	checkCudaErrors(cudaBindTexture(0, edgeIndexesOfTriangleTex, gMemory->dEdgeIndicesOfTriangleTable, channelDescSigned));
	checkCudaErrors(cudaBindTexture(0, numVerticesTex, gMemory->dNumVerticesTable, channelDescUnsigned));
	checkCudaErrors(cudaBindTexture(0, vertexIndexesOfEdgeTex, gMemory->dVertexIndicesOfEdgeTable, channelDescUnsigned));  
}