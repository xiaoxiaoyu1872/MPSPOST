#ifndef FILEDATA_H
#define FILEDATA_H

#include <string>
#include <functional>
#include <vector>
#include <tuple>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>

#include <vtkPoints.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkPolyDataWriter.h>
#include <vtkTriangle.h>

#include "Params.h"
#include "Define.h"

#include "GPUmemory.h"

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;



class FileData{
private:
  int exclude;
  std::string exFile;

  Params *params;
  GPUmemory* gMemory;

  vtkSmartPointer<vtkPoints> Points;
  vtkSmartPointer<vtkFloatArray> Normals;

  vtkSmartPointer< vtkCellArray > triangles;
  vtkSmartPointer<vtkPolyDataWriter> writer;

  std::vector<Vertex> positions;
  std::vector<Normal> normals;
  std::vector<VertexIndex> verindex;

  std::vector<ScalarValue> scalarValue;

  vtkSmartPointer<vtkFloatArray> Rhop;
  vtkSmartPointer<vtkFloatArray> Vel_;
  vtkSmartPointer<vtkFloatArray> Vel;


//-------------------------------diffuse---------------------------------
  vtkSmartPointer<vtkPoints> DiffuseParticles;


 public:
  int frameIndex;

  FileData(Params *params, GPUmemory *gMemory);

  void loadFluidFiledat(int frameIndex);

  void loadBoundFile();
  void loadFluidFile(int frameIndex);

  void saveSurfaceVTKfile();
  void saveSurfacePLYfile();

  void saveDiffuseVTKfile(std::vector<DiffuseParticle> diffuse, uint type);

  void setExclusionZone(std::string const& fileName);

};

#endif
