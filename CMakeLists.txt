# Fewtwo
# copyright 2017 William La Cava 
# license: GNU/GPL v3

# version requirement for Eigen
cmake_minimum_required (VERSION 3.0)
# use c++ 11
set (CMAKE_CXX_STANDARD 11)

# project name
project(Fewtwo)

# add Eigen include paths
find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

# add Shogun includes
find_package(Shogun REQUIRED NO_MODULE)
# find Shogun library
#find_library(Shogun REQUIRED NO_MODULE)

# autocomplete for YouCompleteMe needs commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set a default build type if none was specified
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' since none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

# add OpenMP support
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# gather list of source files
file(GLOB src_files "src/*.cc")

# executable
add_executable(fewtwo ${src_files})

target_link_libraries(fewtwo shogun)
