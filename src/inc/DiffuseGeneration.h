#ifndef DIFFUSEGENERATION_H_
#define DIFFUSEGENERATION_H_

#include "Params.h"
#include "GPUmemory.h"
#include "myCuda.cuh"
#include "FileData.h"

class DiffuseGeneration
{
  public:
      DiffuseGeneration(GPUmemory *gMemory, Params *params, FileData *fileData);
      ~DiffuseGeneration();
      void runsimulation();
  private:
      GPUmemory* gMemory;
      Params* params;
      FileData* fileData;

      uint NumFreeSurfaceParticles;
      uint NumParticles;

      uint GeneratedNumDiffuseParticles;
      uint OldNumDiffuseParticles;

      uint NumIsDiffuseParticles;
      
      std::vector<DiffuseParticle> mDiffuse;

      void constantMemCopy_Sim();
      void constantMemCopy_Diffuse();
      void constantMemCopy_Grid();

      void processingOfFreesurface();
      void estimatingOfPotention();
      void generatingOfDiffuse();
      void updatingOfDiffuse();

      void extractionOfFreeSurfaceParticles();
      void thrustscan_freeparticles();
      void streamcompact_freeparticles();
      void memallocation_freeparticles();
      void transformmatrices_freeparticles();

      void memallocation_potional();
      void calculationOftrappedair();
      void calculationOfwavecrests();

      void calculationOfnumberofdiffuseparticles();
      void thrustscan_diffuseparticles();
      void streamcompact_diffuseparticles();
      void memallocation_diffuseparticles();
      void calculationOfdiffuseposition();
      void determinationOfdiffusetype();
      
      void memallocation_olddiffuseparticles();

      void deleteAndappendParticles();

      void savemiddlefile();
};

#endif 