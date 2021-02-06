cmake_minimum_required(VERSION 3.0)

project(Visualization)

# set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

find_package(CUDA REQUIRED)
find_package(VTK COMPONENTS vtkFiltersModeling vtkIOGeometry REQUIRED)

set(CUDA_NVCC_FLAGS "-arch=sm_61")
set(CUDA_NVCC_FLAGS -O2; -G; -g;-std=c++14)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp-simd")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")


include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc) 
include(${VTK_USE_FILE})

add_subdirectory(src)

