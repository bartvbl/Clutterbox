#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.0)

set(CUDAHOSTCXX /usr/bin/g++-6)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-6)

project (qsiverification CXX CUDA)

OPTION (USE_OpenMP "Use OpenMP" ON)

set (CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_VERBOSE_MAKEFILE 0) # 1 should be used for debugging
set (CMAKE_SUPPRESS_REGENERATION TRUE) # Suppresses ZERO_CHECK

# Compiler Flags

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17 -frounding-math") #  -g -pg -pg -no-pie
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

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_package(CUDA REQUIRED)

INCLUDE(FindCUDA)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=\"sm_61,compute_61\" -lcurand -lineinfo)
set(CUDA_VERBOSE_BUILD ON)

# libShapeSearch
add_subdirectory(../libShapeSearch ${CMAKE_CURRENT_BINARY_DIR}/libSpinImage)


#
# Include Paths
#
include_directories (
  src/
  ../libShapeSearch/src
  lib/glm
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
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)


target_link_libraries (${PROJECT_NAME} ${CUDA_LIBRARY} cuda cudart SpinImage cudadevrt)