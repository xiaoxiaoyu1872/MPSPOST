#include "SurfaceReconstruction.h"
#include "Thrust.cuh"

SurfaceReconstruction::SurfaceReconstruction(GPUmemory *_gMemory, Params *_params, FileData *_fileData)
{   
    params = _params;
    gMemory = _gMemory;
	fileData = _fileData;

	methodOfYu = new MethodOfYu(gMemory, params);
	marchingCube = new MarchingCube(gMemory, params);
}

SurfaceReconstruction::~SurfaceReconstruction()
{   
	delete methodOfYu;
	delete marchingCube;
	std::cout << "~SurfaceReconstruction" << std::endl;
}

void SurfaceReconstruction::runsimulation()
{
	gMemory->SurfaceMemreset();

	methodOfYu->processingOfParticles();
	methodOfYu->processingOfVertices();
	methodOfYu->estimationOfscalarField();

	marchingCube->triangulation();
	fileData->saveSurfaceVTKfile();

	gMemory->MethodYuMemFree();
	gMemory->MCMemFree();
}


void SurfaceReconstruction::saveMiddleFile()
{
	// std::vector<NumParticleGridScan> mNumParticleGridScan;
	// mNumParticleGridScan.resize(params->mGridParams.spSize);

	// std::vector<NumParticleGrid> mNumParticleGrid;
	// mNumParticleGrid.resize(params->mGridParams.spSize);

	// std::vector<IndexRange> mIndexRange;
	// mIndexRange.resize(params->mGridParams.spSize);

	// std::vector<FluidParticle> mFluidParticle;
	// mFluidParticle.resize(gMemory->NumParticles);

	// cudaMemcpy(static_cast<void*>(mNumParticleGridScan.data()),
	// gMemory->dNumSurParticleGridScan, sizeof(NumParticleGridScan) * 
	// params->mGridParams.spSize, cudaMemcpyDeviceToHost);

	// cudaMemcpy(static_cast<void*>(mNumParticleGrid.data()),
	// gMemory->dNumSurParticleGrid, sizeof(NumParticleGrid) * 
	// params->mGridParams.spSize, cudaMemcpyDeviceToHost);


	// std::vector<NumParticleGridScan> mNumInvParticleGridScan;
	// mNumInvParticleGridScan.resize(params->mGridParams.spSize);

	// std::vector<NumParticleGrid> mNumInvParticleGrid;
	// mNumInvParticleGrid.resize(params->mGridParams.spSize);

	// cudaMemcpy(static_cast<void*>(mNumInvParticleGridScan.data()),
	// gMemory->dNumInvParticleGridScan, sizeof(NumParticleGridScan) * 
	// params->mGridParams.spSize, cudaMemcpyDeviceToHost);

	// cudaMemcpy(static_cast<void*>(mNumInvParticleGrid.data()),
	// gMemory->dNumInvParticleGrid, sizeof(NumParticleGrid) * 
	// params->mGridParams.spSize, cudaMemcpyDeviceToHost);


	// cudaMemcpy(static_cast<void*>(mIndexRange.data()),
	// gMemory->dIndexRange, sizeof(IndexRange) * 
	// params->mGridParams.spSize, cudaMemcpyDeviceToHost);

	// cudaMemcpy(static_cast<void*>(mFluidParticle.data()),
	// gMemory->dFluidParticle, sizeof(FluidParticle) * 
	// gMemory->NumParticles, cudaMemcpyDeviceToHost);

	// std::vector<Index> mSurfaceParticlesIndex;
	// mSurfaceParticlesIndex.resize(NumSurfaceParticles);

	// cudaMemcpy(static_cast<void*>(mSurfaceParticlesIndex.data()),
	// gMemory->dSurfaceParticlesIndex, sizeof(Index) * 
	// NumSurfaceParticles, cudaMemcpyDeviceToHost);

	// std::vector<Index> mInvovleParticlesIndex;
	// mInvovleParticlesIndex.resize(NumInvolveParticles);

	// cudaMemcpy(static_cast<void*>(mInvovleParticlesIndex.data()),
	// gMemory->dInvolveParticlesIndex, sizeof(Index) * 
	// NumInvolveParticles, cudaMemcpyDeviceToHost);


	// std::vector<MeanPos> mMeanParticle;
	// mMeanParticle.resize(NumSurfaceParticles);

	// cudaMemcpy(static_cast<void*>(mMeanParticle.data()),
	// gMemory->dSurfaceParticlesMean, sizeof(MeanPos) * 
	// NumSurfaceParticles, cudaMemcpyDeviceToHost);

	// std::vector<SmoothedPos> mSmoothedParticle;
	// mSmoothedParticle.resize(NumSurfaceParticles);

	// cudaMemcpy(static_cast<void*>(mSmoothedParticle.data()),
	// gMemory->dSurfaceParticlesSmoothed, sizeof(SmoothedPos) * 
	// NumSurfaceParticles, cudaMemcpyDeviceToHost);


	// std::vector<SmoothedPos> mSmoothedInvParticle;
	// mSmoothedInvParticle.resize(NumInvolveParticles);

	// cudaMemcpy(static_cast<void*>(mSmoothedInvParticle.data()),
	// gMemory->dInvolveParticlesSmoothed, sizeof(SmoothedPos) * 
	// NumInvolveParticles, cudaMemcpyDeviceToHost);


	// std::vector<Index> mSurfaceVerticeIndex;
	// mSurfaceVerticeIndex.resize(NumSurfaceVertices);

	// cudaMemcpy(static_cast<void*>(mSurfaceVerticeIndex.data()),
	// gMemory->dSurfaceVerticesIndex, sizeof(Index) * 
	// NumSurfaceVertices, cudaMemcpyDeviceToHost);

	// std::vector<ScalarFieldGrid> mScalarFiled;
	// mScalarFiled.resize(params->mGridParams.scSize);

	// cudaMemcpy(static_cast<void*>(mScalarFiled.data()),
	// gMemory->dScalarFiled, sizeof(ScalarFieldGrid) * 
	// params->mGridParams.scSize, cudaMemcpyDeviceToHost);
	

	//---------------------------extraction particles--------------------------
	// std::string basename = "TestParticle";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (long i = 0; i < params->mGridParams.spSize; i++) 
    // {

	// 		// if (mNumParticleGrid[i] == 0)
	// 		// {
	// 		// 	continue;
	// 		// }
			
	// 		for (long index = mIndexRange[i].start; index < mIndexRange[i].end; index++)
	// 		{
	// 			file    << mFluidParticle[index].pos << ' '
	// 					<< mFluidParticle[index].vel << ' '
	// 					<< mFluidParticle[index].rhop << ' '
	// 					<< std::endl;
	// 		}
	// }
    
	//---------------------------sur particles index--------------------------

	// std::string basename = "TestSurfaceIndex";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumSurfaceParticles; i++)
	// {
	// 	int index = mSurfaceParticlesIndex[i];
	// 	file<< mFluidParticle[index].pos << " "
	// 		<< mFluidParticle[index].nor << " "
	// 		<< index 
	// 		<< std::endl;
	// }
	
	//---------------------------sur particles index--------------------------

	// std::string basename = "TestInvIndex";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumInvolveParticles; i++)
	// {
	// 	int index = mInvovleParticlesIndex[i];
	// 	file      << mFluidParticle[index].pos << " "
	// 			  // << mFluidParticle[index].nor << " "
	// 			//   << index 
	// 			  << std::endl;
	// }

	// for (long i = 0; i < params->mGridParams.spSize; i++) 
    // {

	// 		if (mNumInvParticleGrid[i] == 0)
	// 		{
	// 			continue;
	// 		}
			
	// 		for (long index = mIndexRange[i].start; index < mIndexRange[i].end; index++)
	// 		{
	// 			file << mFluidParticle[index].pos << ' '
	// 				//  << mFluidParticle[index].nor << ' '
	// 				 << std::endl;
	// 		}
	// }


	//---------------------------mean particles--------------------------

	// std::string basename = "TestMean";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumSurfaceParticles; i++)
	// {
	// 	file << mMeanParticle[i].pos << std::endl;
	// }

	//---------------------------smoothed particles--------------------------

	// std::string basename = "TestSmoothed";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumSurfaceParticles; i++)
	// {
	// 	file << mSmoothedParticle[i].pos << std::endl;
	// }

	//---------------------------smoothed particles--------------------------

	// std::string basename = "TestSmoothedInv";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);

	// for (int i = 0; i < NumInvolveParticles; i++)
	// {
	// 	file << mSmoothedInvParticle[i].pos << std::endl;
	// }

	//---------------------------extraction vertices--------------------------

	// std::string basename = "TestVer";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);
	// for (long i = 0; i < params->mGridParams.scSize; i++) 
    // {

	// 	if (mIsSurfaceVertices[i] == 0)
	// 	{
	// 		continue;
	// 	}
	// 	float3 vPos = getVertexPos(index1DTo3D(i, params->mGridParams.scresolution),
	// 		params->mGridParams.minPos, params->mGridParams.scGridSize);

	// 	file << vPos << std::endl;
    // }

	//---------------------------vertices compaction--------------------------

	// std::string basename = "TestVerCom_4";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
	// file.open(path.c_str(), std::ios::out);
	// for (long i = 0; i < NumSurfaceVertices; i++) 
    // {
	// 	uint Index = mSurfaceVerticeIndex[i];
	// 	float3 vPos = getVertexPos(index1DTo3D(Index, params->mGridParams.scresolution),
	// 		params->mGridParams.minPos, params->mGridParams.scGridSize);
	// 	file << vPos << ' '<< mScalarFiled[Index] <<
	// 	std::endl;
    // }
}

