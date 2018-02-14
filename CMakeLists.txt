# Feat
# copyright 2017 William La Cava 
# license: GNU/GPL v3

# version requirement for Eigen
cmake_minimum_required (VERSION 3.0)
# use c++ 14
set (CMAKE_CXX_STANDARD 14)

# project name
project(Feat)

# add Eigen include paths
option(EIGEN_DIR "use Eigen environment variable" OFF)
if (EIGEN_DIR)
    include_directories("$ENV{EIGEN3_INCLUDE_DIR}")
else()
    find_package(Eigen3 REQUIRED NO_MODULE)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# add Shogun includes
# find Shogun library
#find_library(Shogun REQUIRED NO_MODULE)
option(SHOGUN_DIR "use environment Shogun paths" OFF)
if (SHOGUN_DIR)
    include_directories("$ENV{SHOGUN_DIR}")
    link_directories("$ENV{SHOGUN_LIB}")
else()
    find_package(Shogun REQUIRED NO_MODULE)
endif()

# autocomplete for YouCompleteMe needs commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set a default build type if none was specified
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' since none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()

# add OpenMP support
option(OMP "use OpenMP" ON)
if (OMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# gather list of source files
file(GLOB_RECURSE src_files "src/*.cc")

# executable
add_executable(feat ${src_files})

target_link_libraries(feat shogun)
