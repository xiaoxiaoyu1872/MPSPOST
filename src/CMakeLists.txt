include_directories(${CMAKE_CURRENT_SOURCE_DIR} ./inc ./inc/marchingcube  ./cuda  ./inc/inireader ./inc/tinyply/)

FILE(GLOB cpp_SOURCES  *.cpp  ./inc/marchingcube/*.cpp)
FILE(GLOB cuda_SOURCES ./cuda/*.cu)

# add_definitions(-std=gnu++11)

add_library(
  Vis
  ${cpp_SOURCES}
)

cuda_add_library(
  cuVis
  ${cuda_SOURCES}
  inc/inireader/ini.c
  inc/inireader/INIReader.cpp
  inc/tinyply/tinyply.cpp
)

add_executable(
    VisGPU
    ${cpp_SOURCES}
)

target_link_libraries(
    VisGPU Vis cuVis -lstdc++fs ${VTK_LIBRARIES} ${CXX_LDFLAGS}
)