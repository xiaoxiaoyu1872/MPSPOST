#include "myCuda.cuh"
#include "GPUmemory.h"
#include "SVD3.cuh"

#include <helper_math.h>

__constant__  SimParams      dSimParams;
__constant__  DiffuseParams  dDiffuseParams;
__constant__  GridParams     dGridParams;

__constant__ uint     dNumFreeSurfaceParticles;
__constant__ uint     dGeneratedNumDiffuseParticles;
__constant__ uint     dOldNumDiffuseParticles;
__constant__ uint     dNumParticles;

__global__
void estimationOfFreeSurfaceParticles
(
	SpatialGrid* spatialGrid,
    BoundGrid *boundGrid,
    NumParticleGrid* numSurParticleGrid,
    IndexRange* particlesIndexRange
) 
{
	uint threadId = getthreadIdGlobal();

	if (threadId >= dGridParams.spSize)
		return;

	if (!spatialGrid[threadId].fluid)
		return;

    // if (boundGrid[threadId].bound)
	// 	return;

    // if (spatialGrid[threadId].bound)
	// 	return;    

	bool isSurface = false;

	uint3 spatialGridRes = dGridParams.spresolution;
	uint3 Index3Dsp = index1DTo3D(threadId, dGridParams.spresolution);

	uint3 lower = make_uint3(Index3Dsp.x - 1, Index3Dsp.y - 1, Index3Dsp.z - 1);
	uint3 upper = make_uint3(Index3Dsp.x + 1, Index3Dsp.y + 1, Index3Dsp.z + 1);
	lower = clamp(lower, make_uint3(0, 0, 0), make_uint3(spatialGridRes.x - 1, spatialGridRes.y - 1, spatialGridRes.z - 1));
	upper = clamp(upper, make_uint3(0, 0, 0), make_uint3(spatialGridRes.x - 1, spatialGridRes.y - 1, spatialGridRes.z - 1));


#pragma unroll 3
	for (uint z = lower.z; z <= upper.z; ++z)
	{
#pragma unroll 3
		for (uint y = lower.y; y <= upper.y; ++y)
		{
#pragma unroll 3
			for (uint x = lower.x; x <= upper.x; ++x)
			{
				uint3 neighbor = make_uint3(x, y, z); 
				uint index = index3DTo1D(neighbor, spatialGridRes);  				
				bool flag = spatialGrid[index].fluid;
				// bool bound = boundGrid[index].bound;
				if (!flag)
				{
					isSurface = true;
					IndexRange indexRange = particlesIndexRange[threadId];

					uint num = indexRange.end - indexRange.start;
					numSurParticleGrid[threadId] = num;
					break;
				}
			}
		}
	}
	
	
	if (!isSurface) 
		return;
	

#pragma unroll 3
	for (uint z = lower.z; z <= upper.z; ++z)
	{
#pragma unroll 3
		for (uint y = lower.y; y <= upper.y; ++y)
		{
#pragma unroll 3
			for (uint x = lower.x; x <= upper.x; ++x)
			{
				uint3 neighbor = make_uint3(x, y, z); 
				uint index = index3DTo1D(neighbor, spatialGridRes);  				
				bool fluid = spatialGrid[index].fluid;
				// bool bound = boundGrid[index].bound;
				// if (!fluid && !bound)
				{
					spatialGrid[threadId].classify = 1;
				}
				
			}
		}
	}
}

__global__
void compactationOfFreeParticles
(
	NumParticleGrid* numParticleGrid, 
	NumParticleGridScan* numParticleGridScan,  
	IndexRange* particlesIndexRange, 
	Index* particlesIndex
) 
{
	uint threadId = getthreadIdGlobal();

	if(threadId >= dGridParams.spSize) 
        return;

    if(numParticleGrid[threadId] > 0)
	{
		uint start = numParticleGridScan[threadId];
		IndexRange range = particlesIndexRange[threadId];
		uint count = range.end - range.start;

		for (uint i = 0; i < count; ++i)
		{			
			particlesIndex[start + i] = range.start + i;
		}
	}
}

__global__
void calculationOfTransformMatricesFree(
	FluidParticle* particles, 
	IndexRange* particlesIndexRange, 
	Index* particlesIndex,
    SpatialGrid* spatialGrid,
    ThinFeature* thinFeature) 
{

	uint threadId = getthreadIdGlobal();

	if (threadId >= dNumFreeSurfaceParticles)
		return;

	uint particleIndex = particlesIndex[threadId];

	float3 pos = particles[particleIndex].pos;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint extent = dGridParams.spexpandExtent_diffuse;
	uint3 minIndex3D = cubeIndex3D - 2 * extent;
	uint3 maxIndex3D = cubeIndex3D + 2 * extent;

	uint3 spatialGridInfo = dGridParams.spresolution;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	MatrixValue cov;
	cov.a11 = cov.a22 = cov.a33 = dDiffuseParams.smoothingRadiusSq;
	cov.a12 = cov.a13 = cov.a21 = cov.a23 = cov.a31 = cov.a32 = 0;

	uint numNeighbors = 0;
	float wSum = 0.0f;
    float3 posMean = {0.0f, 0.0f, 0.0f};
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
					spatialGridInfo)];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 u = neighborPos - pos;

					float dist = sqrt(dot(u, u));
					if (dist <= dDiffuseParams.anisotropicRadius)
						++numNeighbors;
					const float wj = wij(dist, dDiffuseParams.anisotropicRadius);
					wSum += wj;

                    posMean += wj*neighborPos;

					float3 v = neighborPos - pos; 

					cov.a11 += wj * v.x * v.x;
					cov.a22 += wj * v.y * v.y;
					cov.a33 += wj * v.z * v.z;
					float c_12_21 = wj * v.x * v.y;
					float c_13_31 = wj * v.x * v.z;
					float c_23_32 = wj * v.y * v.z;
					cov.a12 += c_12_21;
					cov.a21 += c_12_21;
					cov.a13 += c_13_31;
					cov.a31 += c_13_31;
					cov.a23 += c_23_32;
					cov.a32 += c_23_32;
				}
			}
		}
	}

	float3 n;

	// if (numNeighbors < dDiffuseParams.minNumNeighbors)
	// {
	// 	n.x = 0; n.y = 0; n.z = 0;

	// }
	// else
	{
        posMean /= wSum;
		cov.a11 /= wSum; cov.a12 /= wSum; cov.a13 /= wSum;
		cov.a21 /= wSum; cov.a22 /= wSum; cov.a23 /= wSum;
		cov.a31 /= wSum; cov.a32 /= wSum; cov.a33 /= wSum;

		MatrixValue u;
		float3 v;
		MatrixValue w;

		svd(cov.a11, cov.a12, cov.a13, cov.a21, cov.a22, cov.a23, cov.a31, cov.a32, cov.a33,
			u.a11, u.a12, u.a13, u.a21, u.a22, u.a23, u.a31, u.a32, u.a33,
			v.x, v.y, v.z,
			w.a11, w.a12, w.a13, w.a21, w.a22, w.a23, w.a31, w.a32, w.a33);

		v.x = fabsf(v.x);
		v.y = fabsf(v.y);
		v.z = fabsf(v.z);

        if (v.z < 0.25 * v.x)
		{
			thinFeature[threadId] = 1;
		}

        n.x = u.a13;
		n.y = u.a23;
		n.z = u.a33;	
	}

    if (dot(n, (pos - posMean)) < 0)
    {
        n = -n;
    }

	particles[particleIndex].nor = n;
}


__global__
void calculationofColorField(
	FluidParticle* particles,
	IndexRange* particlesIndexRange,
    ColorField* colorField)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumParticles)
		return;

	float3 pos = particles[threadId].pos;
	float  rhop = particles[threadId].rhop;

	float h = dDiffuseParams.smoothingRadius;
	float mass = dSimParams.mass;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 spatialGridInfo = dGridParams.spresolution; 

	uint3 minIndex3D = cubeIndex3D - dGridParams.spexpandExtent_diffuse;
	uint3 maxIndex3D = cubeIndex3D + dGridParams.spexpandExtent_diffuse;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
					spatialGridInfo)];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					
					float3 sp = pos - neighborPos;
					float mp = length(sp);
                    float q = mp/h;
                    if (q > 0 && q <= 2)
					{
						float ad = 21. / (16. * PI * h * h * h);
						float e1 = (1. - (q / 2.0));
						colorField[threadId] += (mass / rhop) * ad * e1 * e1 * e1 *
                                   e1 * (2 * q + 1.);
					}
				}
			}
		}
	}
}



__global__
void calculationofNormal(
	FluidParticle* particles,
	IndexRange* particlesIndexRange,
    Index* particlesIndex,
    ColorField* colorField)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumFreeSurfaceParticles)
		return;

    uint particleIndex = particlesIndex[threadId];

	float3 pos = particles[particleIndex].pos;

	float h = dDiffuseParams.smoothingRadius;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 spatialGridInfo = dGridParams.spresolution; 

	uint3 minIndex3D = cubeIndex3D - dGridParams.spexpandExtent_diffuse;
	uint3 maxIndex3D = cubeIndex3D + dGridParams.spexpandExtent_diffuse;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
					spatialGridInfo)];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					
					float3 sp = pos - neighborPos;
					float mp = length(sp);
                    float q = mp/h;

					if (q > 0 && q <= 2)
					{
						float ad = 21. / (16. * PI * h * h * h);
	                    float e1 = (1. - (q / 2.0));
                        float rval = colorField[i] * ad * e1 * e1 * e1 * e1 * (2 * q + 1.);

                       particles[particleIndex].nor += rval * sp;
					}
				}
			}
		}
	}
}


__global__
void calculationofNormal(
	FluidParticle* particles,
	IndexRange* particlesIndexRange,
    Index* particlesIndex)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumFreeSurfaceParticles)
		return;

    uint particleIndex = particlesIndex[threadId];

	float3 pos = particles[particleIndex].pos;

	float h = dDiffuseParams.smoothingRadius;
	float h2 = dDiffuseParams.smoothingRadiusSq;

	float coefficient =  dDiffuseParams.coefficient;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 spatialGridInfo = dGridParams.spresolution; 

	uint3 minIndex3D = cubeIndex3D - dGridParams.spexpandExtent_diffuse;
	uint3 maxIndex3D = cubeIndex3D + dGridParams.spexpandExtent_diffuse;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	float3 normal = {0,0,0};
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
					spatialGridInfo)];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					
					float3 sp = pos - neighborPos;
					float mp2 = dot(sp,sp);
					float mp = sqrtf(mp2);
                    float q = mp/h;

					if (q > 0 && q < 1)
					{

						// float coefficient = (4 / 3)*PI*powf((dSimParams.particleSpacing/ 2), 3) 
						// * 315 / (64 * PI*powf(h, 9));

						float derivative = powf((h2 - mp2), 2) * 6; 

                        normal += coefficient * sp / mp * derivative;
					}
				}
			}
		}
	}

	float mn2 = dot(normal,normal);
	if (mn2 < EPSILON_)
	{
		return;
	}

	float invLen = rsqrtf(mn2);
	particles[particleIndex].nor = invLen * normal;
}



__global__
void calculationofWavecrests(
	FluidParticle* particles,
	IndexRange* particlesIndexRange,
	Index* particlesIndex,
    ThinFeature* thinFeature,
    DiffusePotential* diffusePotential)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumFreeSurfaceParticles)
		return;

	uint particleIndex = particlesIndex[threadId];

	float3 pos = particles[particleIndex].pos;
	float3 vel = particles[particleIndex].vel;
	float3 nor = particles[particleIndex].nor;

	bool flag = thinFeature[threadId];

	if (dot(nor,nor) <= EPSILON_ || dot(vel,vel) <= EPSILON_)
	{
		return;
	}

    if (dot(vel,nor) < 0.6)
	{
        return;
	}

	float h = dDiffuseParams.smoothingRadius;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 spatialGridInfo = dGridParams.spresolution; 

    uint extend = dGridParams.spexpandExtent_diffuse;

	uint3 minIndex3D = cubeIndex3D - extend;
	uint3 maxIndex3D = cubeIndex3D + extend;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				uint index1D = index3DTo1D(index3D, spatialGridInfo);
				IndexRange indexRange = particlesIndexRange[index1D];
				
				if (indexRange.start == 0xffffffff)
					continue;

				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 neighborNor = particles[i].nor;
                    if (dot(neighborNor, neighborNor) <= EPSILON_)
                        continue;
					
                    diffusePotential[particleIndex].waveCrest += 
                    crests2p(pos, neighborPos, nor, neighborNor, h, flag); 
				}
			}
		}
	}
}



__global__
void calculationofTrappedairpotential(
	FluidParticle* particles,
	IndexRange* particlesIndexRange,
    DiffusePotential* diffusePotential)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumParticles)
		return;

	float3 pos = particles[threadId].pos;
	float3 vel = particles[threadId].vel;

	float h = dDiffuseParams.smoothingRadius;
	float mass = dSimParams.mass;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint3 spatialGridInfo = dGridParams.spresolution; 

	uint3 minIndex3D = cubeIndex3D - dGridParams.spexpandExtent_diffuse;
	uint3 maxIndex3D = cubeIndex3D + dGridParams.spexpandExtent_diffuse;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));


	// int neighbor_num = 0;
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
					spatialGridInfo)];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 neighborVel = particles[i].vel;
					
					float3 sp = pos - neighborPos;
					float mp = length(sp);

					if (mp > EPSILON_ && mp <= h)
					{
						float3 sv = vel - neighborVel;
						float mv = length(sv);

						if (mv > 0)
						{
                            float q = mp/h;
							float3 dv = sv/mv;
							float3 dp = sp/mp;
							
							float e = 1.0 - dot(dv,dp);
							float w = 1.0 - q;

							diffusePotential[threadId].Ita += mv * e * w; 
                        
						}
					}

				}
			}
		}
	}

	// if(neighbor_num > 25)
	diffusePotential[threadId].energy = 0.5 * mass * dot(vel,vel); 
}

__global__
void calculateofNumberofdiffuseparticles(
	DiffusePotential* diffusePotential,
	NumDiffuseParticle* numDiffuseParticle,
	IsDiffuse* isDiffuse)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dNumParticles)
		return;

	diffusePotential[threadId].waveCrest = phi(diffusePotential[threadId].waveCrest, 
    dDiffuseParams.minWaveCrests, dDiffuseParams.maxWaveCrests);

	diffusePotential[threadId].Ita = phi(diffusePotential[threadId].Ita, 
    dDiffuseParams.minTrappedAir, dDiffuseParams.maxTrappedAir);

	diffusePotential[threadId].energy = phi(diffusePotential[threadId].energy, 
    dDiffuseParams.minKineticEnergy, dDiffuseParams.maxKineticEnergy);

	numDiffuseParticle[threadId] = floor(diffusePotential[threadId].energy * 
    (dDiffuseParams.trappedAirMultiplier * diffusePotential[threadId].Ita + 
    dDiffuseParams.waveCrestsMultiplier * diffusePotential[threadId].waveCrest) * dDiffuseParams.timeStep);
	
    if (numDiffuseParticle[threadId] > 0)
	{
		isDiffuse[threadId] = 1;
	}
}


__global__
void compactationOfDiffuseParticle(
	NumDiffuseParticle* numDiffuseParticle,
	NumDiffuseParticleScan*  numDiffuseParticleScan,
	Index* diffuseParticlesIndex)
{
	uint threadId = getthreadIdGlobal();

	if (threadId >= dNumParticles) 
        return;
    
    if (numDiffuseParticle[threadId] > 0)
	{
		diffuseParticlesIndex[numDiffuseParticleScan[threadId]] = threadId;
	}
	
}


__global__
void calculateofDiffusePosition(
	FluidParticle* particles,
	NumDiffuseParticle* numDiffuseParticle,
	NumDiffuseParticleScan* numDiffuseParticleScan,
	DiffuseParticle* diffuseParticle,
	float* tempRand,
	DiffusePotential* diffusePotential,
	Index* diffuseParticlesIndex)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dGeneratedNumDiffuseParticles)
		return;

	uint particleIndex = diffuseParticlesIndex[threadId];

	float3 pos = particles[particleIndex].pos;
	float3 vel = particles[particleIndex].vel;

	float sp_h = dDiffuseParams.smoothingRadius;

	// Obtain orthogonal vectors to velocity vector
	float3 e1, e2;

	if (vel.x != 0) 
	{ // x non zero

      e1.x = solveEq(pos.z, pos.y, pos.x, vel.z, vel.y, vel.x, 0, 1);
      e1.y = 1;
      e1.z = 0;
      e1 = normalize(e1);
     
    } else if (vel.y != 0) 
    { // y non zero

      e1.x = 1; 
      e1.z = 0;
      e1.y = solveEq(pos.x, pos.z, pos.y, vel.x, vel.z, vel.y, 1, 0);
      e1 = normalize(e1);
    } else 
    { // z non zero

      e1.x = 1;
      e1.y = 0;
      e1.z = solveEq(pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, 1, 0);	
      e1 = normalize(e1);
    }


    e2.x = e1.y * vel.z - vel.y * e1.z;
    e2.y = e1.x * vel.z - vel.x * e1.z;
    e2.z = e1.x * vel.y - vel.x * e1.y;

    e2 = normalize(e2);

    float3 nvel = normalize(vel);

    uint idif = 0;
	uint start = numDiffuseParticleScan[particleIndex];

	for (int j = 0; j < numDiffuseParticle[particleIndex]; j++) 
	{

		idif = start + j;

	    float h = tempRand[idif * 3] * (length(vel) * dDiffuseParams.timeStep) * 0.5;
	    
	    float r = sp_h * sqrt(tempRand[idif * 3 + 1]); 
	    
	    float theta = tempRand[idif * 3 + 2] * 2 * PI;

	    diffuseParticle[idif].pos.x = pos.x + r * cos(theta) * e1.x + r * sin(theta) * e2.x + h * nvel.x;
	    diffuseParticle[idif].pos.y = pos.y + r * cos(theta) * e1.y + r * sin(theta) * e2.y + h * nvel.y,
	    diffuseParticle[idif].pos.z = pos.z + r * cos(theta) * e1.z + r * sin(theta) * e2.z + h * nvel.z;

	    diffuseParticle[idif].vel.x = r * cos(theta) * e1.x + r * sin(theta) * e2.x + vel.x;
	    diffuseParticle[idif].vel.y = r * cos(theta) * e1.y + r * sin(theta) * e2.y + vel.y;
	    diffuseParticle[idif].vel.z = r * cos(theta) * e1.z + r * sin(theta) * e2.z + vel.z;

	    diffuseParticle[idif].TTL = numDiffuseParticle[particleIndex] * dDiffuseParams.lifeTime;

	    diffuseParticle[idif].life = 1;

	    if (diffuseParticle[idif].pos > dGridParams.maxPos_diffuse || diffuseParticle[idif].pos < dGridParams.minPos_diffuse)
	    {
	    	diffuseParticle[idif].life = 0;
	    }

		float3 aa = {1, -1, -1};
		float3 bb = {7, 2, 2};
		
		if (diffuseParticle[idif].pos.x > aa.x  &&  diffuseParticle[idif].pos.y > aa.y && diffuseParticle[idif].pos.z > aa.z 
			&&
			diffuseParticle[idif].pos.x < bb.x && diffuseParticle[idif].pos.y < bb.y && diffuseParticle[idif].pos.y < bb.y)
		{
			diffuseParticle[idif].life = 0;
		}
 	}

}


__global__
void calculateofDiffuseType(
	FluidParticle* particles,
	IndexRange* particlesIndexRange,
	DiffuseParticle* diffuseParticle,
	SpatialGrid* spatialGrid)
{
	uint threadId = getthreadIdGlobal();
	if (threadId >= dGeneratedNumDiffuseParticles)
		return;

	float3 pos = diffuseParticle[threadId].pos;

	uint3 Index3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	uint index1D = index3DTo1D(Index3D, dGridParams.spresolution);

	uint3 minIndex3D = Index3D - dGridParams.spexpandExtent_diffuse;
	uint3 maxIndex3D = Index3D + dGridParams.spexpandExtent_diffuse;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(dGridParams.spresolution.x - 1,
		dGridParams.spresolution.y - 1, dGridParams.spresolution.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(dGridParams.spresolution.x - 1,
		dGridParams.spresolution.y - 1, dGridParams.spresolution.z - 1));

	diffuseParticle[threadId].type = spatialGrid[index1D].classify;

	if (diffuseParticle[threadId].type != 1)
	{
		return;
	}

	int density = 0;
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
					dGridParams.spresolution)];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 sp = pos - neighborPos;
					float mp = length(sp);

					if (mp <= dDiffuseParams.smoothingRadius && mp > 0)
					{
						density++;
					}
				}
			}
		}
	}
	if (density > BUBBLE)
	{
		diffuseParticle[threadId].type = 2;
	}
	else if (density < SPARY)
	{
		diffuseParticle[threadId].type = 0;
	}
	else
	{
		diffuseParticle[threadId].type = 1;
	}
}


__global__
void updateDiffuseParticle(
	FluidParticle* particles,
	IndexRange* particlesIndexRange,
	DiffuseParticle* diffuseParticle,
	SpatialGrid* spatialGrid)
{

	uint threadId = getthreadIdGlobal();
	if (threadId >= dOldNumDiffuseParticles)
		return;

	float3 pos = diffuseParticle[threadId].pos;

	float3 vel = diffuseParticle[threadId].vel;

	float h = dDiffuseParams.smoothingRadius;

    uint3 spatialGridInfo = dGridParams.spresolution;

	uint3 cubeIndex3D = getIndex3D(pos, dGridParams.minPos, dGridParams.spGridSize);

	int index1D = index3DTo1D(cubeIndex3D, spatialGridInfo);

    diffuseParticle[threadId].type = spatialGrid[index1D].classify;

	uint3 minIndex3D = cubeIndex3D - dGridParams.spexpandExtent_diffuse;
	uint3 maxIndex3D = cubeIndex3D + dGridParams.spexpandExtent_diffuse;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(dGridParams.spresolution.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	int density;
	density = 0;
	if (diffuseParticle[threadId].type == 1)
	{
		for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
		{
			for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
			{
				for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
				{
					uint3 index3D = make_uint3(xSp, ySp, zSp);
					IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
						spatialGridInfo)];
					if (indexRange.start == 0xffffffff)
						continue;
					for (uint i = indexRange.start; i < indexRange.end; i++)
					{
						float3 neighborPos = particles[i].pos;
						float3 sp = pos - neighborPos;
						float mp = length(sp);
						if (mp <= dDiffuseParams.smoothingRadius && mp > 0)
						{
							density++;
						}
					}
				}
			}
		}

		if (density > BUBBLE)
		{
			diffuseParticle[threadId].type = 2;
		}
		else if (density < SPARY)
		{
			diffuseParticle[threadId].type = 0;
		}
		else
		{
			diffuseParticle[threadId].type = 1;
		}
	}
	
    float wij = 0;
    float3 num = {0, 0, 0};

	// This is not needed for spray particles.
	if (diffuseParticle[threadId].type > 0.5)
	{
		for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
		{
			for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
			{
				for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
				{
					uint3 index3D = make_uint3(xSp, ySp, zSp);
					IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
						spatialGridInfo)];
					if (indexRange.start == 0xffffffff)
						continue;
					for (uint i = indexRange.start; i < indexRange.end; i++)
					{
						float3 neighborPos = particles[i].pos;
						float tval = Wwendland(pos - neighborPos, h);
						num = num + particles[i].vel * tval;
						wij += tval;
					}
				}
			}
		}

	}

	float3 g;
	g.x = 0;
	g.y = 0;
	g.z = -9.81;

	///////////////////!!!!!!!!!!!!!!!!!!!!! 注意重力的方向！！！！！！！！！！！！！！！
	// Now we can re-clasify and calculate new positions
	if (diffuseParticle[threadId].type == 0)  //it is spary
	{
		diffuseParticle[threadId].vel.x = diffuseParticle[threadId].vel.x + g.x * dDiffuseParams.timeStep;
		diffuseParticle[threadId].vel.y = diffuseParticle[threadId].vel.y + g.y * dDiffuseParams.timeStep;
        diffuseParticle[threadId].vel.z = diffuseParticle[threadId].vel.z + g.z * dDiffuseParams.timeStep;
		
		diffuseParticle[threadId].pos = diffuseParticle[threadId].pos + dDiffuseParams.timeStep * diffuseParticle[threadId].vel;
		
		diffuseParticle[threadId].type = 0;
	}

	else if(diffuseParticle[threadId].type == 2)
	{
		num = num/wij;
		diffuseParticle[threadId].vel.x = vel.x + dDiffuseParams.timeStep * ((-dDiffuseParams.buoyancyControl) * (g.x) + dDiffuseParams.dragControl * (num.x - vel.x) / dDiffuseParams.timeStep);
		diffuseParticle[threadId].vel.z = vel.z + dDiffuseParams.timeStep * ((-dDiffuseParams.buoyancyControl) * (g.z) + dDiffuseParams.dragControl * (num.z - vel.z) / dDiffuseParams.timeStep);
		diffuseParticle[threadId].vel.y = vel.y + dDiffuseParams.timeStep * ((-dDiffuseParams.buoyancyControl) * (g.y) + dDiffuseParams.dragControl * (num.y - vel.y) / dDiffuseParams.timeStep);

		diffuseParticle[threadId].pos = diffuseParticle[threadId].pos + dDiffuseParams.timeStep * diffuseParticle[threadId].vel;
		diffuseParticle[threadId].type = 2;
	}

	else
	{
		num = num/wij;
		diffuseParticle[threadId].vel = num;
		diffuseParticle[threadId].pos = diffuseParticle[threadId].pos + dDiffuseParams.timeStep * num;

		diffuseParticle[threadId].type = 1;
		diffuseParticle[threadId].TTL--;
		if (diffuseParticle[threadId].TTL <= 0)
		{
			diffuseParticle[threadId].life = 0;
			return;
		}
	}

	if (diffuseParticle[threadId].pos > dGridParams.maxPos_diffuse || diffuseParticle[threadId].pos < dGridParams.minPos_diffuse)
	{
		diffuseParticle[threadId].life = 0;
		return;
	}

	float3 aa = {1, -1, -1};
	float3 bb = {7, 2, 2};
	
	if (diffuseParticle[threadId].pos.x > aa.x  &&  diffuseParticle[threadId].pos.y > aa.y && diffuseParticle[threadId].pos.z > aa.z 
		&&
		diffuseParticle[threadId].pos.x < bb.x && diffuseParticle[threadId].pos.y < bb.y && diffuseParticle[threadId].pos.y < bb.y)
	{
		diffuseParticle[threadId].life = 0;
		return;
	}

	float3 newpos = diffuseParticle[threadId].pos;

	cubeIndex3D = getIndex3D(newpos, dGridParams.minPos, dGridParams.spGridSize);

	index1D = index3DTo1D(cubeIndex3D, spatialGridInfo);

	diffuseParticle[threadId].type = spatialGrid[index1D].classify;

	if (diffuseParticle[threadId].type != 1)
	{
		return;
	}

	minIndex3D = cubeIndex3D - dGridParams.spexpandExtent_diffuse;
	maxIndex3D = cubeIndex3D + dGridParams.spexpandExtent_diffuse;

	minIndex3D = clamp(minIndex3D, make_uint3(0, 0, 0), make_uint3(spatialGridInfo.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));
	maxIndex3D = clamp(maxIndex3D, make_uint3(0, 0, 0), make_uint3(dGridParams.spresolution.x - 1,
		spatialGridInfo.y - 1, spatialGridInfo.z - 1));

	density = 0;
	for (int zSp = minIndex3D.z; zSp <= maxIndex3D.z; zSp++)
	{
		for (int ySp = minIndex3D.y; ySp <= maxIndex3D.y; ySp++)
		{
			for (int xSp = minIndex3D.x; xSp <= maxIndex3D.x; xSp++)
			{
				uint3 index3D = make_uint3(xSp, ySp, zSp);
				IndexRange indexRange = particlesIndexRange[index3DTo1D(index3D,
					spatialGridInfo)];
				if (indexRange.start == 0xffffffff)
					continue;
				for (uint i = indexRange.start; i < indexRange.end; i++)
				{
					float3 neighborPos = particles[i].pos;
					float3 sp = newpos - neighborPos;
					float mp = length(sp);

					if (mp <= dDiffuseParams.smoothingRadius && mp > 0)
					{
						density++;
					}
				}
			}
		}
	}
	if (density > BUBBLE)
	{
		diffuseParticle[threadId].type = 2;
	}
	else if (density < SPARY)
	{
		diffuseParticle[threadId].type = 0;
	}
	else
	{
		diffuseParticle[threadId].type = 1;
	}

}