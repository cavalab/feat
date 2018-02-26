# Feat
# copyright 2017 William La Cava 
# license: GNU/GPL v3

######### Options ###########################
option( CORE_USE_CUDA   "Use CUDA to speed up certain parts of the code."  ON )
#############################################

######### CUDA decisions ####################
if (CORE_USE_CUDA)
    MESSAGE( STATUS ">> --------------  USING CUDA --------------" )
    set( CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0")
    if (APPLE OR UNIX)
        set(CMAKE_C_COMPILER   /usr/bin/gcc)
        set(CMAKE_CXX_COMPILER /usr/bin/g++)
    elseif (WIN32)
        ### By default we use VS
        MESSAGE( STATUS ">> User compiler:             MSVC" )
        MESSAGE( STATUS ">> Choosing a different compiler is not yet implemented for Windows" )
    endif()
endif()
#############################################

# version requirement for Eigen
cmake_minimum_required (VERSION 3.0)
# use c++ 14
set (CMAKE_CXX_STANDARD 14)

set (CMAKE_CXX_FLAGS "--std=c++11")

set (CUDA_PROPAGATE_HOST_FLAGS ON)

  
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

######### CUDA decisions ####################
if (CORE_USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    #enable_language(CUDA)

endif()
#############################################

# gather list of source files
if (CORE_USE_CUDA)   
    file(GLOB_RECURSE cuda_files "*.cu")
    include_directories("src/node-cuda")
else()
    include_directories("src/node/")
endif()    

file(GLOB_RECURSE src_main_files "src/main.cc")
file(GLOB_RECURSE src_cv_files "src/cv_main.cc")

# executable
if (CORE_USE_CUDA)
    cuda_add_executable(feat ${src_main_files} ${cuda_files})
    cuda_add_executable(feat_cv ${src_cv_files} ${cuda_files})
    #set_property(TARGET feat PROPERTY CUDA_STANDARD 14)
    #set_property(TARGET feat_cv PROPERTY CUDA_STANDARD 14)

    target_link_libraries(feat shogun ${CUDA_LIBRARIES})
    target_link_libraries(feat_cv shogun ${CUDA_LIBRARIES})
else()
    add_executable(feat ${src_main_files})
    add_executable(feat_cv ${src_cv_files})

    target_link_libraries(feat shogun)
    target_link_libraries(feat_cv shogun)
endif()
