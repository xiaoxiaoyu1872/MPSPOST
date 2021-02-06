#include <omp.h>

#include "DiffuseGeneration.h"
#include "DiffuseGeneration_kernel.cu"
#include "Thrust.cuh"

clock_t start;
clock_t finish; 
double duration;

void getduration()
{
    duration = (double) (finish- start)/CLOCKS_PER_SEC; 
    printf("%f\n", duration);
}

DiffuseGeneration::DiffuseGeneration(GPUmemory *_gMemory, Params *_params, FileData *_fileData)
{
    params = _params;
    gMemory = _gMemory;
	fileData = _fileData;
    
    OldNumDiffuseParticles = 0;

    constantMemCopy_Sim();
	constantMemCopy_Diffuse();
	constantMemCopy_Grid();

    gMemory->DiffuseAlloFixedMem();

    NumParticles = gMemory->NumParticles;

}

DiffuseGeneration::~DiffuseGeneration()
{
    gMemory->DiffuseFreeFixedMem();
    gMemory->Memfree_bound();
    gMemory->DiffuseMemFinal();
}


void DiffuseGeneration::constantMemCopy_Sim()
{
	checkCudaErrors(cudaMemcpyToSymbol(dSimParams, &params->mSimParams, sizeof(SimParams)));
}

void DiffuseGeneration::constantMemCopy_Diffuse()
{
	checkCudaErrors(cudaMemcpyToSymbol(dDiffuseParams, &params->mDiffuseParams, sizeof(DiffuseParams)));
}

void DiffuseGeneration::constantMemCopy_Grid()
{
	checkCudaErrors(cudaMemcpyToSymbol(dGridParams, &params->mGridParams, sizeof(GridParams)));
}


void DiffuseGeneration::runsimulation()
{
    gMemory->DiffuseMemreset();

    start = clock();
    processingOfFreesurface();
    finish = clock();
    printf("processingOfFreesurface consuming : ");
    getduration();

    start = clock();
    estimatingOfPotention(); //!!!!!
    finish = clock();
    printf("estimatingOfPotention consuming : ");
    getduration();

    start = clock();
    generatingOfDiffuse();
    finish = clock();
    printf("generatingOfDiffuse consuming : ");
    getduration();

    start = clock();
    updatingOfDiffuse();
    finish = clock();
    printf("updatingOfDiffuse consuming : ");
    getduration();


    start = clock();
    deleteAndappendParticles();
    finish = clock();
    printf("deleteAndappendParticles consuming : ");
    getduration();

    // savemiddlefile();
    gMemory->DiffuseMemFree();
}

void DiffuseGeneration::processingOfFreesurface()
{
    start = clock();
    extractionOfFreeSurfaceParticles();
    finish = clock();
    printf("FreeSurface consuming : ");
    getduration();


    start = clock();
    thrustscan_freeparticles();
    finish = clock();
    printf("scan consuming : ");
    getduration();

    start = clock();
    memallocation_freeparticles();
    finish = clock();
    printf("mem consuming : ");
    getduration();

    start = clock();
    streamcompact_freeparticles();
    finish = clock();
    printf("compact consuming : ");
    getduration();


    start = clock();
    transformmatrices_freeparticles();
    finish = clock();
    printf("property consuming : ");
    getduration();
}

void DiffuseGeneration::estimatingOfPotention()
{
    start = clock();
    memallocation_potional();
    finish = clock();
    printf("memallocation_potional consuming : ");
    getduration();

    start = clock();
    calculationOftrappedair();
    finish = clock();
    printf("trappedair consuming : ");
    getduration();

    start = clock();
    calculationOfwavecrests();
    finish = clock();
    printf("wavecrests consuming : ");
    getduration();

    cudaDeviceSynchronize();
}

void DiffuseGeneration::generatingOfDiffuse()
{
    calculationOfnumberofdiffuseparticles();
    thrustscan_diffuseparticles();
    memallocation_diffuseparticles();
    streamcompact_diffuseparticles();
    calculationOfdiffuseposition();
    determinationOfdiffusetype();
}


void DiffuseGeneration::extractionOfFreeSurfaceParticles()
{
    uint spSize = params->mGridParams.spSize;
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(spSize, gridDim, blockDim); 

    estimationOfFreeSurfaceParticles <<< gridDim, blockDim >>> (
		gMemory->dSpatialGrid,
        gMemory->dBoundGrid,
		gMemory->dNumFreeSurParticleGrid,
		gMemory->dIndexRange);
}

void DiffuseGeneration::thrustscan_freeparticles()
{
    NumFreeSurfaceParticles = ThrustExclusiveScan(
    gMemory->dNumFreeSurParticleGridScan,
    gMemory->dNumFreeSurParticleGrid,
    (uint)params->mGridParams.spSize);

    if (NumFreeSurfaceParticles == 0)
	{
		std::cerr << "No free surface particle detected!\n";
		return;
	}
    gMemory->NumFreeSurfaceParticles = NumFreeSurfaceParticles;

    std::cout << "mNumFreeSurfaceParticles =  " << NumFreeSurfaceParticles << std::endl;

	std::cout << "free surface particles ratio: " << 
	static_cast<double>(NumFreeSurfaceParticles)
		/ gMemory->NumParticles << std::endl;
}

void DiffuseGeneration::streamcompact_freeparticles()
{
    uint spSize = params->mGridParams.spSize;
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(spSize, gridDim, blockDim); 

   compactationOfFreeParticles << < gridDim, blockDim>> > (
		gMemory->dNumFreeSurParticleGrid,
		gMemory->dNumFreeSurParticleGridScan,
		gMemory->dIndexRange,
		gMemory->dFreeSurfaceParticlesIndex);	
	cudaDeviceSynchronize();
}

void DiffuseGeneration::transformmatrices_freeparticles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumFreeSurfaceParticles, gridDim, blockDim); 

    // calculationOfTransformMatricesFree << < gridDim, blockDim>> > (
	// 	gMemory->dFluidParticle,
	// 	gMemory->dIndexRange,
	// 	gMemory->dFreeSurfaceParticlesIndex,
    //     gMemory->dSpatialGrid,
    //     gMemory->dThinFeature);	
	// cudaDeviceSynchronize();

    // calcGridDimBlockDim(NumParticles, gridDim, blockDim); 
    // calculationofColorField << < gridDim, blockDim>> > (
	// 	gMemory->dFluidParticle,
	// 	gMemory->dIndexRange,
    //     gMemory->dColorField);	
	// cudaDeviceSynchronize();

    // calcGridDimBlockDim(NumFreeSurfaceParticles, gridDim, blockDim); 
    // calculationofNormal << < gridDim, blockDim>> > (
	// 	gMemory->dFluidParticle,
	// 	gMemory->dIndexRange,
	// 	gMemory->dFreeSurfaceParticlesIndex,
    //     gMemory->dColorField);	
	// cudaDeviceSynchronize();

    calcGridDimBlockDim(NumFreeSurfaceParticles, gridDim, blockDim); 
    calculationofNormal << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dFreeSurfaceParticlesIndex);	
	cudaDeviceSynchronize();
}

void DiffuseGeneration::calculationOfwavecrests()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumFreeSurfaceParticles, gridDim, blockDim); 
    calculationofWavecrests << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
		gMemory->dFreeSurfaceParticlesIndex,
        gMemory->dThinFeature,
        gMemory->dDiffusePotential);	
    cudaDeviceSynchronize();
}

void DiffuseGeneration::calculationOftrappedair()
{
    dim3 gridDim, blockDim;
    NumParticles = gMemory->NumParticles;

	calcGridDimBlockDim(NumParticles, gridDim, blockDim); 
    calculationofTrappedairpotential << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
		gMemory->dIndexRange,
        gMemory->dDiffusePotential);	
    cudaDeviceSynchronize();
}


void DiffuseGeneration::calculationOfnumberofdiffuseparticles()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumParticles, gridDim, blockDim); 

    calculateofNumberofdiffuseparticles << < gridDim, blockDim>> > (
		gMemory->dDiffusePotential,
		gMemory->dNumDiffuseParticle,
        gMemory->dIsDiffuse);	
    cudaDeviceSynchronize();
}



void DiffuseGeneration::thrustscan_diffuseparticles()
{

    GeneratedNumDiffuseParticles = ThrustExclusiveScan(
    gMemory->dNumDiffuseParticleScan,
    gMemory->dNumDiffuseParticle,
    (uint)NumParticles);


    NumIsDiffuseParticles = ThrustExclusiveScan(
    gMemory->dIsDiffuseScan,
    gMemory->dIsDiffuse,
    (uint)NumParticles);

    gMemory->GeneratedNumDiffuseParticles = GeneratedNumDiffuseParticles;

    std::cout << "GeneratedNumDiffuseParticles =  " << GeneratedNumDiffuseParticles << std::endl;

    std::cout << "NumIsDiffuseParticles =  " << NumIsDiffuseParticles << std::endl;

	std::cout << "Isdiffuse particles ratio: " << static_cast<double>(NumIsDiffuseParticles)
		/ NumParticles << std::endl;
    
    cudaDeviceSynchronize();
}



void DiffuseGeneration::streamcompact_diffuseparticles()
{   
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumParticles, gridDim, blockDim); 
    compactationOfDiffuseParticle << < gridDim, blockDim>> > (
		gMemory->dNumDiffuseParticle,
		gMemory->dNumDiffuseParticleScan,
        gMemory->dDiffuseParticlesIndex);	
    cudaDeviceSynchronize();
}



void DiffuseGeneration::calculationOfdiffuseposition()
{
    std::random_device rd; //随机数生产器
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> xunif(0, 1);	//0-1

	std::vector<float> tempRand(GeneratedNumDiffuseParticles * 3);
	for (auto &x : tempRand)
		x = xunif(gen);

	float* dtempRand;

	checkCudaErrors(cudaMalloc((void**)&dtempRand, GeneratedNumDiffuseParticles * 3 * sizeof(float)));

	checkCudaErrors(cudaMemcpy(dtempRand, static_cast<void*>(tempRand.data()),
			GeneratedNumDiffuseParticles * 3 * sizeof(float), cudaMemcpyHostToDevice));

	std::vector<float>().swap(tempRand);

    dim3 gridDim, blockDim;
	calcGridDimBlockDim(NumParticles, gridDim, blockDim); 
    calculateofDiffusePosition << < gridDim, blockDim>> > (
        gMemory->dFluidParticle,
		gMemory->dNumDiffuseParticle,
		gMemory->dNumDiffuseParticleScan,
        gMemory->dDiffuseParticle,
        dtempRand,
        gMemory->dDiffusePotential,
        gMemory->dDiffuseParticlesIndex);	
    cudaDeviceSynchronize();

    safeCudaFree((void**)&dtempRand);
}


void DiffuseGeneration::determinationOfdiffusetype()
{
    dim3 gridDim, blockDim;
	calcGridDimBlockDim(GeneratedNumDiffuseParticles, gridDim, blockDim); 
    calculateofDiffuseType << < gridDim, blockDim>> > (
		gMemory->dFluidParticle,
        gMemory->dIndexRange,
		gMemory->dDiffuseParticle,
        gMemory->dSpatialGrid);	
    cudaDeviceSynchronize();
}

void DiffuseGeneration::updatingOfDiffuse()
{
    if (OldNumDiffuseParticles > 0)
    {
        memallocation_olddiffuseparticles();
        dim3 gridDim, blockDim;
        calcGridDimBlockDim(OldNumDiffuseParticles, gridDim, blockDim); 
        updateDiffuseParticle << < gridDim, blockDim>> > (
            gMemory->dFluidParticle,
            gMemory->dIndexRange,
            gMemory->dDiffuseParticle_old,
            gMemory->dSpatialGrid);	
        cudaDeviceSynchronize();
    }
}


void DiffuseGeneration::savemiddlefile()
{
    std::vector<FluidParticle> mFluidParticle;
	mFluidParticle.resize(gMemory->NumParticles);

    cudaMemcpy(static_cast<void*>(mFluidParticle.data()),
	gMemory->dFluidParticle, sizeof(FluidParticle) * 
	gMemory->NumParticles, cudaMemcpyDeviceToHost);

    std::vector<Index> mSurfaceParticlesIndex;
	mSurfaceParticlesIndex.resize(NumFreeSurfaceParticles);

	cudaMemcpy(static_cast<void*>(mSurfaceParticlesIndex.data()),
	gMemory->dFreeSurfaceParticlesIndex, sizeof(Index) * 
	NumFreeSurfaceParticles, cudaMemcpyDeviceToHost);


    std::vector<DiffusePotential> mDiffusePotential;
	mDiffusePotential.resize(gMemory->NumParticles);

    cudaMemcpy(static_cast<void*>(mDiffusePotential.data()),
	gMemory->dDiffusePotential, sizeof(DiffusePotential) * 
	gMemory->NumParticles, cudaMemcpyDeviceToHost);


    // //---------------------------free surface particles--------------------------
    std::vector<DiffuseParticle> mSurfaceParticle;
    mSurfaceParticle.resize(NumFreeSurfaceParticles);
    // std::string basename = "TestFreeSur";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
    // std::ofstream file;
    // file.open(path.c_str(), std::ios::out);

    for (int i = 0; i < NumFreeSurfaceParticles; i++)
	{
		int index = mSurfaceParticlesIndex[i];
        mSurfaceParticle[i].pos = mFluidParticle[index].pos;
        mSurfaceParticle[i].vel = mFluidParticle[index].vel;

		// file<< mFluidParticle[index].pos << " "
		// 	<< mFluidParticle[index].nor << " "
		// 	<< index 
		// 	<< std::endl;
	}

    fileData->saveDiffuseVTKfile(mSurfaceParticle, 3);

    // //---------------------------particle potential-----------------------------
    // std::vector<DiffuseParticle> mSurfaceParticle;
    mSurfaceParticle.resize(NumParticles);

    for (int i = 0; i < NumParticles; i++)
	{
        mSurfaceParticle[i].pos = mFluidParticle[i].pos;
        mSurfaceParticle[i].vel = mFluidParticle[i].vel;
        // mSurfaceParticle[i].nor= mFluidParticle[i].nor;
        mSurfaceParticle[i].energy = mDiffusePotential[i].energy;
        mSurfaceParticle[i].Ita = mDiffusePotential[i].Ita;
        mSurfaceParticle[i].waveCrest = mDiffusePotential[i].waveCrest;

	}
    // fileData->saveDiffuseVTKfile(mSurfaceParticle, 4);

    ConfigParams mConfigParams = params->mConfigParams;
    int frameIndex = fileData->frameIndex;

    std::string basename = "TestPotional";
    std::string seqnum(mConfigParams.nzeros, '0');
    std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
    std::sprintf(&seqnum[0], formats.c_str(), frameIndex);

	std::string path = params->mConfigParams.boundPath + std::string(basename)+ seqnum +  ".dat";
    std::ofstream file;
    file.open(path.c_str(), std::ios::out);

    for (int i = 0; i < NumParticles; i++)
	{
        int index = i;
		file << mFluidParticle[index].pos << " "
            //  << mFluidParticle[index].nor << " "
             << mSurfaceParticle[index].energy << " "
             << mSurfaceParticle[index].Ita << " "
             << mSurfaceParticle[index].waveCrest << " "
			 << std::endl;
	}


    // for (int i = 0; i < NumFreeSurfaceParticles; i++)
	// {
	// 	int index = mSurfaceParticlesIndex[i];
    //     file << mFluidParticle[index].pos << ' ' 
    //          << mFluidParticle[index].nor <<' '
    //          << mSurfaceParticle[index].waveCrest << std::endl;
	// }

}

void DiffuseGeneration::deleteAndappendParticles()
{
    std::vector<DiffuseParticle> gDiffuseParticle;
    std::vector<DiffuseParticle> oDiffuseParticle;

    std::vector<DiffuseParticle> mSpary;
    std::vector<DiffuseParticle> mFoam;
    std::vector<DiffuseParticle> mBubble;

    mDiffuse.clear();
    //--------------------------------------------------------------------------------------------------------

	gDiffuseParticle.resize(GeneratedNumDiffuseParticles); 
	cudaMemcpy(static_cast<void*>(gDiffuseParticle.data()),
	gMemory->dDiffuseParticle, sizeof(DiffuseParticle) * GeneratedNumDiffuseParticles, cudaMemcpyDeviceToHost);
	
    oDiffuseParticle.resize(OldNumDiffuseParticles);
    if (OldNumDiffuseParticles)
    {
        cudaMemcpy(static_cast<void*>(oDiffuseParticle.data()),
	    gMemory->dDiffuseParticle_old, sizeof(DiffuseParticle) * OldNumDiffuseParticles, cudaMemcpyDeviceToHost);
    }

	std::copy(oDiffuseParticle.begin(), oDiffuseParticle.end(), std::back_inserter(gDiffuseParticle));

    mDiffuse.resize(GeneratedNumDiffuseParticles + OldNumDiffuseParticles);
    mSpary.resize(GeneratedNumDiffuseParticles + OldNumDiffuseParticles);
    mFoam.resize(GeneratedNumDiffuseParticles + OldNumDiffuseParticles);
    mBubble.resize(GeneratedNumDiffuseParticles + OldNumDiffuseParticles);

	uint mNumTotalDiffuseParticle = 0;

    uint mNumSparyParticle = 0;
    uint mNumFoamParticle = 0;
    uint mNumBubbleParticle = 0;

	int before = gDiffuseParticle.size();
    
#pragma omp parallel for schedule(guided)
	for (int i = 0; i < gDiffuseParticle.size(); ++i)
	{
		if (!gDiffuseParticle[i].life)
		{
			continue;
		}

		if (gDiffuseParticle[i].type == 0)
		{
			mSpary[mNumSparyParticle] = gDiffuseParticle[i];
            mNumSparyParticle++;
		}
		else if(gDiffuseParticle[i].type == 2)
		{
			mBubble[mNumBubbleParticle] = gDiffuseParticle[i];
            mNumBubbleParticle++;
		}
		else
		{
			mFoam[mNumFoamParticle] = gDiffuseParticle[i];
            mNumFoamParticle++;
		}
		mDiffuse[mNumTotalDiffuseParticle] = gDiffuseParticle[i];
		mNumTotalDiffuseParticle++;
	}

	int after = mNumTotalDiffuseParticle;
    mDiffuse.resize(mNumTotalDiffuseParticle);
    mSpary.resize(mNumSparyParticle);
    mFoam.resize(mNumFoamParticle);
    mBubble.resize(mNumBubbleParticle);

	OldNumDiffuseParticles = mNumTotalDiffuseParticle;
    gMemory->OldNumDiffuseParticles = mNumTotalDiffuseParticle;

    gMemory->memallocation_olddiffuseparticles(mDiffuse);
    
	std::cout << "deleted: " << before - after << std::endl;
	std::cout << "generated = "<< GeneratedNumDiffuseParticles << std::endl;
    std::cout << "total = "<< mNumTotalDiffuseParticle << std::endl;

    std::cout << "mSpary = "<< mSpary.size() << std::endl;
    std::cout << "mBubble = "<< mBubble.size() << std::endl;
    std::cout << "mFoam = "<< mFoam.size() << std::endl;

    fileData->saveDiffuseVTKfile(mSpary, 0);
    fileData->saveDiffuseVTKfile(mFoam, 1);
    fileData->saveDiffuseVTKfile(mBubble, 2);

    std::vector<DiffuseParticle>().swap(gDiffuseParticle);
    std::vector<DiffuseParticle>().swap(oDiffuseParticle);

    std::vector<DiffuseParticle>().swap(mSpary);
    std::vector<DiffuseParticle>().swap(mFoam);
    std::vector<DiffuseParticle>().swap(mBubble);

}



void DiffuseGeneration::memallocation_freeparticles()
{
    gMemory->memallocation_freeparticles();
    checkCudaErrors(cudaMemcpyToSymbol(dNumFreeSurfaceParticles, 
	&gMemory->NumFreeSurfaceParticles, sizeof(uint)));
}

void DiffuseGeneration::memallocation_potional()
{
    gMemory->memallocation_potional();
    checkCudaErrors(cudaMemcpyToSymbol(dNumParticles, 
	&gMemory->NumParticles, sizeof(uint)));
}

void DiffuseGeneration::memallocation_diffuseparticles()
{
    gMemory->memallocation_diffuseparticles();
    checkCudaErrors(cudaMemcpyToSymbol(dGeneratedNumDiffuseParticles, 
	&gMemory->GeneratedNumDiffuseParticles, sizeof(uint)));
}

void DiffuseGeneration::memallocation_olddiffuseparticles()
{
    gMemory->memallocation_olddiffuseparticles(mDiffuse);
    checkCudaErrors(cudaMemcpyToSymbol(dOldNumDiffuseParticles, 
	&gMemory->OldNumDiffuseParticles, sizeof(uint)));
}