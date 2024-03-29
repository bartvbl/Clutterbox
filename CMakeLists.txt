#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.0)

if(!MSVC)
set(CUDAHOSTCXX "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
endif()

project (clutterbox CXX CUDA)

OPTION (USE_OpenMP "Use OpenMP" ON)

set (CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_VERBOSE_MAKEFILE 0) # 1 should be used for debugging
set (CMAKE_SUPPRESS_REGENERATION TRUE) # Suppresses ZERO_CHECK

# PCL

find_package(PCL 1.8.1 REQUIRED PATHS /usr NO_DEFAULT_PATH COMPONENTS common io)

# Compiler Flags

if(MSVC)
    add_definitions("-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /openmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native -frounding-math") #  -g -pg -pg -no-pie
endif()

# OpenMP

IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()

# CUDA

find_package(CUDA REQUIRED)

INCLUDE(FindCUDA)

set(CUDA_NVCC_FLAGS -arch=sm_61 -gencode arch=compute_61,code=\"sm_61,compute_61\" -lcurand -lineinfo -std=c++14)
set(CUDA_VERBOSE_BUILD OFF)

# libShapeSearch
add_subdirectory(../libShapeDescriptor ${CMAKE_CURRENT_BINARY_DIR}/libShapeDescriptor)


#
# Include Paths
#
include_directories (
        src/
        ../libShapeDescriptor/src
        tools/clutterEstimator
        lib/glm
        lib/json
        lib/arrrgh
        lib/Eigen
        lib/nvidia
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)


#
# Add files
#
file (GLOB_RECURSE PROJECT_HEADERS src/*.hpp
        src/*.h
        src/*.cuh)
file (GLOB_RECURSE PROJECT_SOURCES src/*.cpp
        src/*.cxx
        src/*.cc
        src/*.c
        src/*.cu)
file (GLOB         PROJECT_CONFIGS CMakeLists.txt
        README.rst
        .gitignore
        .gitmodules)
file (GLOB_RECURSE CLUTTER_ESTIMATOR_SOURCES    tools/clutterEstimator/*.cpp
        tools/clutterEstimator/*.cu
        tools/clutterEstimator/*.cuh)
file (GLOB_RECURSE CLUTTER_EXPERIMENT_SOURCES
        src/clutterbox/*.cpp
        src/clutterbox/*.cu
        src/utilities/*.cpp)

#
# Organizing files
#
source_group ("headers" FILES ${PROJECT_HEADERS})
source_group ("sources" FILES ${PROJECT_SOURCES})

#
# Set executable and target link libraries

add_executable (${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS}
        ${PROJECT_SHADERS} ${PROJECT_CONFIGS}
        ${VENDORS_SOURCES})


target_link_libraries (${PROJECT_NAME} ${CUDA_LIBRARY} cuda cudart ShapeDescriptor cudadevrt)

add_executable(clutterEstimator ${CLUTTER_ESTIMATOR_SOURCES} ${CLUTTER_EXPERIMENT_SOURCES})
target_link_libraries(clutterEstimator ${CUDA_LIBRARY} cuda cudart ShapeDescriptor cudadevrt)

add_executable(quicciDistanceFunctionBenchmark
        tools/quicciDistanceFunctionComparison/main.cpp
        tools/quicciDistanceFunctionComparison/quicciDistanceFunctionBenchmark.cpp
        tools/quicciDistanceFunctionComparison/clutterSphereMeshAugmenter.cpp
        src/utilities/randomFileSelector.cpp
        src/utilities/listDir.cpp
        src/utilities/stringUtils.cpp
        src/clutterbox/clutterBoxUtilities.cpp)
target_link_libraries(quicciDistanceFunctionBenchmark ShapeDescriptor)