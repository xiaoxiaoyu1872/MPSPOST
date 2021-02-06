#ifndef DEFINE_H_
#define DEFINE_H_

#include "Helper.h"

#define PI 3.14159265f
#define EPSILON_ 1.0e-7
#define SIGMA 1.56668147f //315.0f / 64.0f * 3.1415926535898f

#define SPARY 4
#define BUBBLE 32

#define SURFACE 15

#define MAXUINT 4294967295

typedef unsigned char uchar;

typedef uchar   BOOL;

typedef uchar NumParticleGrid;	
typedef uint   NumParticleGridScan;	

typedef BOOL   IsSurface;
typedef uint   IsSurfaceScan;

typedef BOOL  IsValid;
typedef uint  IsValidScan;

typedef BOOL  IsDiffuse;
typedef uint  IsDiffuseScan;

typedef ushort NumDiffuseParticle;	
typedef uint   NumDiffuseParticleScan;	


typedef uchar  NumVertexCube;
typedef uint   NumVertexCubeScan;

typedef uchar  CubeFlag;

typedef uint   Index;

typedef float  ScalarFieldGrid;

typedef float3 Vertex;
typedef float3 Normal;
typedef uint3  VertexIndex;

typedef float  ColorField;

typedef bool ThinFeature;

struct SimParams
{
    float particleSpacing;    
    float mass;

};

struct SurfaceParams
{
    int  smoothingRadiusRatio;
    float smoothingRadius;
    float smoothingRadiusInv;
    float smoothingRadiusSq;
    float anisotropicRadius;

    int minNumNeighbors;
    int isolateNumNeighbors;

    float lambdaForSmoothed;

    float isoValue;
};


struct DiffuseParams
{
    int  smoothingRadiusRatio;
    float smoothingRadius;
    float smoothingRadiusInv;
    float smoothingRadiusSq;
    float anisotropicRadius;

    char minNumNeighbors;

    float coefficient;

    float minWaveCrests;
    float maxWaveCrests;

    float minTrappedAir;
    float maxTrappedAir;

    float minKineticEnergy;
    float maxKineticEnergy;

    float buoyancyControl;
    float dragControl;

    int trappedAirMultiplier;
    int waveCrestsMultiplier;

    int lifeTime;

    float timeStep;
};


struct GridParams
{
    float3 minPos;
    float3 maxPos;

    int spScale;
    int scScale;
    float sptoscScaleInv;
    
    float spGridSize;
    float scGridSize;

    uint3 spresolution;
    uint3 scresolution;

    int spexpandExtent;
    int scexpandExtent;

    //------------------------------------diffuse-----------------------------------
    float3 minPos_diffuse;
    float3 maxPos_diffuse;

    // float spGridSize_diffuse;
    // uint3 spresolution_diffuse;

    int spexpandExtent_diffuse;

    uint spSize;
    uint scSize;
};


struct ConfigParams
{
    int gpu_id;

    int frameStart;
    int frameEnd;
    int frameStep;

    bool isDiffuse;
    bool isSurface;
    std::string Directory_Param;
    std::string FileName_Param;

    int nzeros;
    std::string fluidPath;
    std::string fluidPrefix;
    
    std::string boundPath;
    std::string boundPrefix;

    std::string surfacePrefix;
    std::string surfacePath;

    std::string diffusePrefix;
    std::string diffusePath;
};


struct FluidParticle
{ 
	float3 pos; 
	float3 vel;
    float3 nor;
    float  rhop;
};


struct SmoothedPos
{ 
	float3 pos; 
};

struct MeanPos
{ 
	float3 pos; 
};

struct DiffuseParticle
{ 
	float3 pos; 
    float3 vel;
    char type;
    uchar TTL;
    bool life;

    float waveCrest;
    float Ita;
    float energy;
};


struct DiffusePotential
{ 
	float waveCrest;
    float Ita;
    float energy;
};


struct BoundParticle
{ 
    float3 pos; 
};

struct BoundGrid
{
	bool bound;
};

struct SpatialGrid
{
    bool fluid;
    bool inner;
    bool bound;
    char classify;
};

struct ScalarValue
{
    float3 vel;
    float vel_;
    float rhop;
};

struct MatrixValue 
{
	float a11, a12, a13;
	float a21, a22, a23;
	float a31, a32, a33;
	float maxvalue;
};

struct IndexRange { uint start, end;};

#endif