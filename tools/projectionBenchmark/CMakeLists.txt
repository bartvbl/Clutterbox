#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.0)

project (projectionBenchmark CXX)

OPTION (USE_OpenMP "Use OpenMP" ON)

set (CMAKE_CXX_STANDARD 14)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_VERBOSE_MAKEFILE 0) # 1 should be used for debugging
set (CMAKE_SUPPRESS_REGENERATION TRUE) # Suppresses ZERO_CHECK

# PCL

find_package(PCL 1.8.1 REQUIRED COMPONENTS common io)

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

add_executable(projectionBenchmark compare.cpp)
target_link_libraries(projectionBenchmark ${PCL_LIBRARIES})
