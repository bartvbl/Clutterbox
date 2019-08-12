cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(compare)

find_package(PCL 1.8.1 REQUIRED COMPONENTS common io)
find_package (Eigen3 3.3.4 REQUIRED NO_MODULE)


set (CMAKE_CXX_STANDARD 17)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(compare compare.cpp)
target_link_libraries(compare ${PCL_LIBRARIES} Eigen3::Eigen)
