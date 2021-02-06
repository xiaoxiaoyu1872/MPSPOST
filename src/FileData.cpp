#include "FileData.h"
#include <string>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkPolyDataReader.h>
#include <vtkCommand.h>

#include <array>

#include "tinyply.h"
using namespace tinyply;

class ErrorObserver : public vtkCommand
{
public:
  ErrorObserver():
    Error(false),
    Warning(false),
    ErrorMessage(""),
    WarningMessage("") {}

  static ErrorObserver *New()
  {
    return new ErrorObserver;
  }

  bool GetError() const
  {
    return this->Error;
  }

  bool GetWarning() const
  {
    return this->Warning;
  }

  void Clear()
  {
    this->Error = false;
    this->Warning = false;
    this->ErrorMessage = "";
    this->WarningMessage = "";
  }

  virtual void Execute(vtkObject *vtkNotUsed(caller),
                       unsigned long event,
                       void *calldata)
  {
  switch(event)
    {
    case vtkCommand::ErrorEvent:
      ErrorMessage = static_cast<char *>(calldata);
      this->Error = true;
      break;
    case vtkCommand::WarningEvent:
      WarningMessage = static_cast<char *>(calldata);
      this->Warning = true;
      break;
    }
  }

  std::string GetErrorMessage()
  {
    return ErrorMessage;
  }

  std::string GetWarningMessage()
  {
    return WarningMessage;
  }

private:
  bool        Error;
  bool        Warning;
  std::string ErrorMessage;
  std::string WarningMessage;
};

FileData::FileData(Params *_params, GPUmemory *_gMemory)
{
    params = _params;
    gMemory = _gMemory;
}

void FileData::setExclusionZone(std::string const &fileName) 
{
  exclude = true;
  exFile = fileName;
}

void FileData::loadFluidFile(int _frameIndex) 
{
    frameIndex = _frameIndex;
    ConfigParams mConfigParams = params->mConfigParams;
    std::string seqnum(mConfigParams.nzeros, '0');
    std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
    std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
    std::string fileName = (fs::path(mConfigParams.fluidPath) / 
    (mConfigParams.fluidPrefix + seqnum + ".vtk")).generic_string();

    std::cout << "Opening: " << fileName << std::endl;
    std::cout << "\n\n== [" << " Step " << frameIndex << " of " << mConfigParams.frameEnd 
    << " ] ===================================================================\n";

    vtkSmartPointer<ErrorObserver> errorObserver =
      vtkSmartPointer<ErrorObserver>::New();
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();

    reader->SetFileName(fileName.c_str());
    reader->AddObserver(vtkCommand::ErrorEvent, errorObserver);
    reader->Update();

    if (errorObserver->GetError())
    {
        std::cerr << "ERROR: the file cannot be loaded." << std::endl;
    }

    vtkPolyData *output = reader->GetOutput();

    vtkDataArray *Points = output->GetPoints()->GetData();

    vtkPointData *pointData = output->GetPointData();

    vtkDataArray *pvel = pointData->GetArray("Vel");
    vtkDataArray *rhop = pointData->GetArray("Rhop"); 

    std::vector<FluidParticle> mFluidParticle;
    mFluidParticle.resize(output->GetPoints()->GetNumberOfPoints());

    for (long i = 0; i < output->GetPoints()->GetNumberOfPoints(); i++) 
    {
        double *p = Points->GetTuple(i);
        double *v = pvel->GetTuple(i);

        FluidParticle pi;
        pi.pos = {float(p[0]), float(p[1]), float(p[2])};
        pi.vel = {float(v[0]), float(v[1]), float(v[2])};
        pi.rhop = float(rhop->GetTuple(i)[0]);

        mFluidParticle[i] = pi;
    }

    gMemory->memAlcandCpy_fluid(mFluidParticle);

    std::vector<FluidParticle>().swap(mFluidParticle);
}



int readnum(std::string infilename) {
    int numPoints = 0;  
    std::string tmp;

    ifstream infile_line; 
    infile_line.open(infilename);

    if (infile_line.fail())  
    {  
        std::cout<<"Could not open"<<infilename<<std::endl;
    }  

    while(getline(infile_line,tmp))
    {       
        numPoints++;
    }

    return numPoints;
}

void FileData::loadFluidFiledat(int frameIndex)
{
  ConfigParams mConfigParams = params->mConfigParams;
  std::string infilename = "pdata." +  std::to_string(frameIndex) + ".dat";
  infilename = (fs::path(mConfigParams.fluidPath)/infilename);   

  int numPoints = readnum(infilename);

  float x,y,z;
  float vx,vy,vz;
  float rhop;

  std::ifstream in(infilename, std::ios::in);
  std::cout<<"numPoints = "<< numPoints <<std::endl;
  std::cout<<"frameIndex = "<< frameIndex <<std::endl;

  std::vector<FluidParticle> mFluidParticle;
  mFluidParticle.resize(numPoints);

  for (int i=0; i < numPoints; i++) {
      in  >> x >> y >> z
          >> vx >> vy >> vz 
          >> rhop;
      FluidParticle pi;

      pi.pos.x = x;
      pi.pos.y = y;
      pi.pos.z = z;

      pi.vel.x = x;
      pi.vel.y = y;
      pi.vel.z = z;

      pi.rhop = 1000;

      mFluidParticle[i] = pi;
  }

  gMemory->memAlcandCpy_fluid(mFluidParticle);

  std::vector<FluidParticle>().swap(mFluidParticle);
}



void FileData::loadBoundFile()
{
    ConfigParams mConfigParams = params->mConfigParams;
    std::string fileName = (fs::path(mConfigParams.boundPath) / 
    (mConfigParams.boundPrefix + "Bound" + ".vtk")).generic_string();

    std::cout << "Opening: " << fileName << std::endl;

    vtkSmartPointer<ErrorObserver> errorObserver =
      vtkSmartPointer<ErrorObserver>::New();
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();

    reader->SetFileName(fileName.c_str());
    reader->AddObserver(vtkCommand::ErrorEvent, errorObserver);
    reader->Update();

    if (errorObserver->GetError())
    {
        std::cerr << "ERROR: the file cannot be loaded." << std::endl;
    }

    vtkPolyData *output = reader->GetOutput();

    vtkDataArray *Points = output->GetPoints()->GetData();

    vtkPointData *pointData = output->GetPointData();

    vtkDataArray *type = pointData->GetArray("Type"); 

    std::vector<BoundParticle> mBoundParticle;
    mBoundParticle.resize(output->GetPoints()->GetNumberOfPoints());

    int NumBoundParticles = 0;
    for (long i = 0; i < output->GetPoints()->GetNumberOfPoints(); i++) 
    {
        double *p = Points->GetTuple(i);

        BoundParticle pi;
        pi.pos = {float(p[0]), float(p[1]), float(p[2])};

        if (type->GetTuple(i)[0] < 0.5)  //0: Bound 1: Piston 2: floatingBox
        {
            mBoundParticle[NumBoundParticles] = pi;
            NumBoundParticles++;
        }
    }
    
    gMemory->memAlcandCpy_bound(mBoundParticle);

    std::vector<BoundParticle>().swap(mBoundParticle);
}

void FileData::saveSurfaceVTKfile()
{
  size_t nums = gMemory->NumSurfaceMeshVertices;

  positions.resize(nums);
  normals.resize(nums);
  scalarValue.resize(nums);

  checkCudaErrors(cudaMemcpy(static_cast<void*>(positions.data()), gMemory->dVertex,
		sizeof(Vertex) * nums, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(static_cast<void*>(normals.data()), gMemory->dNormal,
		sizeof(Normal) * nums, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(static_cast<void*>(scalarValue.data()), gMemory->dScalarValue,
		sizeof(ScalarValue) * nums, cudaMemcpyDeviceToHost));

	std::cout << "Number of vertices : " << nums << std::endl;
	std::cout << "Number of face: " << nums/3 << std::endl;

  Points = vtkSmartPointer<vtkPoints>::New();
  Points->SetNumberOfPoints(nums);

  Normals = vtkSmartPointer<vtkFloatArray>::New();
  Normals->SetName("Normal");
  Normals->SetNumberOfComponents(3);
  Normals->SetNumberOfTuples(nums);

  Rhop = vtkSmartPointer<vtkFloatArray>::New();
  Rhop->SetName("Rhop");
  Rhop->SetNumberOfComponents(1);
  Rhop->SetNumberOfTuples(nums);

  Vel_ = vtkSmartPointer<vtkFloatArray>::New();
  Vel_->SetName("VelLength");
  Vel_->SetNumberOfComponents(1);
  Vel_->SetNumberOfTuples(nums);

  Vel = vtkSmartPointer<vtkFloatArray>::New();
  Vel->SetName("Vel");
  Vel->SetNumberOfComponents(3);
  Vel->SetNumberOfTuples(nums);

  for (int i = 0; i < nums; i++)
	{
    auto p =  positions[i];
    auto n =  normals[i];
    auto rhop = scalarValue[i].rhop;
    auto vel_ = scalarValue[i].vel_;
    auto vel = scalarValue[i].vel;
		Points->InsertPoint(static_cast<vtkIdType>(i), p.x, p.y, p.z);
    Normals->InsertTuple3(static_cast<vtkIdType>(i), n.x , n.y, n.z);

    Rhop->InsertTuple1(static_cast<vtkIdType>(i), rhop);
    Vel_->InsertTuple1(static_cast<vtkIdType>(i), vel_);
    Vel->InsertTuple3(static_cast<vtkIdType>(i), vel.x, vel.y, vel.z);

	}

  triangles =  vtkSmartPointer< vtkCellArray >::New();

  for (int i = 0; i < nums; i += 3)
	{
    vtkSmartPointer< vtkTriangle > triangle = vtkSmartPointer< vtkTriangle >::New();
    for (int j = 0; j < 3; j++)
    {
       triangle->GetPointIds()->SetId(j, i + j);
    }
    triangles->InsertNextCell(triangle);
	}

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

  polydata->SetPoints(Points);
  polydata->SetPolys(triangles);
  // polydata->GetPointData()->AddArray(Normals);

  // polydata->GetPointData()->AddArray(Rhop);
  polydata->GetPointData()->AddArray(Vel_);
  // polydata->GetPointData()->AddArray(Vel);

  vtkSmartPointer<vtkPolyDataWriter> writer =
      vtkSmartPointer<vtkPolyDataWriter>::New();

  frameIndex = frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.surfacePath) / 
  (mConfigParams.surfacePrefix + seqnum + ".vtk")).generic_string();

  // writer->SetFileTypeToBinary();
  writer->SetFileTypeToASCII();
  writer->SetInputData(polydata);

  writer->SetFileName(fileName.c_str());
  writer->Write();
}






void FileData::saveDiffuseVTKfile(std::vector<DiffuseParticle> _diffuse, uint type)
{  
  std::vector<DiffuseParticle> diffuse = _diffuse;

  DiffuseParticles = vtkSmartPointer<vtkPoints>::New();
  DiffuseParticles->SetNumberOfPoints(diffuse.size());

  vtkSmartPointer<vtkFloatArray> velocity = vtkSmartPointer<vtkFloatArray>::New();
  velocity->SetName("Velocity");
  velocity->SetNumberOfComponents(3);
  velocity->SetNumberOfTuples(diffuse.size());

  vtkSmartPointer<vtkFloatArray> size = vtkSmartPointer<vtkFloatArray>::New();
  size->SetName("Size");
  size->SetNumberOfComponents(1);
  size->SetNumberOfTuples(diffuse.size());

  vtkSmartPointer<vtkFloatArray> energy = vtkSmartPointer<vtkFloatArray>::New();
  energy->SetName("energy");
  energy->SetNumberOfComponents(1);
  energy->SetNumberOfTuples(diffuse.size());

  vtkSmartPointer<vtkFloatArray> Ita = vtkSmartPointer<vtkFloatArray>::New();
  Ita->SetName("Ita");
  Ita->SetNumberOfComponents(1);
  Ita->SetNumberOfTuples(diffuse.size());

  vtkSmartPointer<vtkFloatArray> waveCrest = vtkSmartPointer<vtkFloatArray>::New();
  waveCrest->SetName("waveCrest");
  waveCrest->SetNumberOfComponents(1);
  waveCrest->SetNumberOfTuples(diffuse.size());

  for (int i = 0; i < diffuse.size(); i++)
	{
    auto pos =  diffuse[i].pos;
    auto vel =  diffuse[i].vel;

		DiffuseParticles->InsertPoint(static_cast<vtkIdType>(i), pos.x, pos.y, pos.z);
    velocity->InsertTuple3(static_cast<vtkIdType>(i), vel.x , vel.y, vel.z);
    size->InsertTuple1(static_cast<vtkIdType>(i), params->mDiffuseParams.smoothingRadius/10);

    energy->InsertTuple1(static_cast<vtkIdType>(i), diffuse[i].energy);
    Ita->InsertTuple1(static_cast<vtkIdType>(i), diffuse[i].Ita);
    waveCrest->InsertTuple1(static_cast<vtkIdType>(i), diffuse[i].waveCrest);
	}

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(DiffuseParticles);

  vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
  for (long i = 0; i < diffuse.size(); ++i){
    vtkIdType pt[] = {i};
    vertices->InsertNextCell(1, pt);
  }
  polydata->SetVerts(vertices);
  polydata->GetPointData()->AddArray(velocity);
  polydata->GetPointData()->AddArray(size);

  polydata->GetPointData()->AddArray(energy);
  polydata->GetPointData()->AddArray(Ita);
  polydata->GetPointData()->AddArray(waveCrest);

  vtkSmartPointer<vtkPolyDataWriter> writer =
      vtkSmartPointer<vtkPolyDataWriter>::New();

  writer->SetFileTypeToBinary();
  writer->SetInputData(polydata);

  frameIndex = frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;
  if (type == 0)
    mConfigParams.diffusePrefix = "PartSpary_";
  if (type == 1)
    mConfigParams.diffusePrefix = "PartFoam_";
  if (type == 2)
    mConfigParams.diffusePrefix = "PartBubble_";  

  if (type == 3)
    mConfigParams.diffusePrefix = "PartSurface_";

  if (type == 4)
    mConfigParams.diffusePrefix = "PartDebug_";
  
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.diffusePath) / 
  (mConfigParams.diffusePrefix + seqnum + ".vtk")).generic_string();

  writer->SetFileName(fileName.c_str());
  writer->Write();
}

void FileData::saveSurfacePLYfile()
{
	size_t nums = gMemory->NumSurfaceMeshVertices;

	std::vector<Vertex> positions;
	std::vector<Normal> normals;
	std::vector<VertexIndex> verindex;

	positions.resize(nums);
	normals.resize(nums);
	verindex.resize(nums/3);

	if (nums == 0)
	{
		std::cerr << "Nothing produced.\n";
		return;
	}

	for (int i = 0; i < nums; i = i + 3)
	{
		verindex[i/3].x = i;
		verindex[i/3].y = i + 1;
		verindex[i/3].z = i + 2;
	}

	checkCudaErrors(cudaMemcpy(static_cast<void*>(positions.data()), gMemory->dVertex,
		sizeof(Vertex) * nums, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(static_cast<void*>(normals.data()), gMemory->dNormal,
		sizeof(Normal) * nums, cudaMemcpyDeviceToHost));

	std::cout << "tinyply " << std::endl;
	std::cout << "Number of vertices : " << nums << std::endl;
	std::cout << "Number of face: " << nums/3 << std::endl;


	std::string basename = "TestSurfaceR";
	std::string path = params->mConfigParams.boundPath + std::string(basename) + ".ply";
    std::ofstream file(path.c_str());


	PlyFile file_ply;

    file_ply.add_properties_to_element("vertex", { "x", "y", "z" }, 
        Type::FLOAT32, positions.size(), reinterpret_cast<uint8_t*>(positions.data()), Type::INVALID, 0);

	file_ply.add_properties_to_element("face", { "vertex_indices" },
        Type::UINT32, verindex.size(), reinterpret_cast<uint8_t*>(verindex.data()), Type::UINT8, 3);

    file_ply.get_comments().push_back("generated by tinyply 2.3");

	file_ply.write(file, true);
}

