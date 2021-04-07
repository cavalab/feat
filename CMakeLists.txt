# Feat
# copyright 2017 William La Cava 
# license: GNU/GPL v3

######### Options ###########################
option( CORE_USE_CUDA   "Use CUDA to speed up certain parts of the code."  OFF )
#############################################

######### CUDA decisions ####################
#if (CORE_USE_CUDA)
#    MESSAGE( STATUS ">> --------------  USING CUDA --------------" )
#    set( CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0")
#    if (APPLE OR UNIX)
#        set(CMAKE_C_COMPILER   /usr/bin/gcc)
#        set(CMAKE_CXX_COMPILER /usr/bin/g++)
#    elseif (WIN32)
#        ### By default we use VS
#        MESSAGE( STATUS ">> User compiler:             MSVC" )
#        MESSAGE( STATUS ">> Choosing a different compiler is not yet implemented for Windows" )
#    endif()
#endif()
#############################################

# version requirement for Eigen
cmake_minimum_required (VERSION 3.0)
# use c++ 14
set (CMAKE_CXX_STANDARD 14)

#set (CMAKE_CXX_FLAGS "--std=c++11")

#set (CUDA_PROPAGATE_HOST_FLAGS ON)

  
# project name
project(Feat)

# add Eigen include paths
if (DEFINED ENV{EIGEN3_INCLUDE_DIR})
    set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
    include_directories($ENV{EIGEN3_INCLUDE_DIR})
elseif(DEFINED EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
else()
    message(STATUS "No EIGEN3_INCLUDE_DIR found, using find_package...")
    find_package(Eigen3 REQUIRED NO_MODULE)
endif()
message(STATUS "EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")

# add Shogun includes
# find Shogun library
#find_library(Shogun REQUIRED NO_MODULE)
if (DEFINED ENV{SHOGUN_DIR})
    set(SHOGUN_DIR $ENV{SHOGUN_DIR})
    set(SHOGUN_LIB $ENV{SHOGUN_LIB})
    include_directories($ENV{SHOGUN_DIR})
    link_directories($ENV{SHOGUN_LIB})
elseif(DEFINED SHOGUN_DIR AND DEFINED SHOGUN_LIB)
    include_directories(${SHOGUN_DIR})
    link_directories(${SHOGUN_LIB})
else()
    message(STATUS "No SHOGUN_DIR/SHOGUN_LIB found, using find_package...")
    find_package(Shogun REQUIRED NO_MODULE)
endif()
message(STATUS "SHOGUN_DIR=${SHOGUN_DIR}")
message(STATUS "SHOGUN_LIB=${SHOGUN_LIB}")

# Locate GTest
option(GTEST "build tests" OFF)
if (GTEST)
    option(GTEST_DIR "use GTest environment path" OFF)
    if (GTEST_DIR)
        include_directories("$ENV{GTEST_DIR}/include")

    else()
        message(STATUS "Searching for GTest using find_package...")
        find_package(GTest REQUIRED)
        include_directories(${GTEST_INCLUDE_DIRS})
    endif()
endif()

# autocomplete for YouCompleteMe needs commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set a default build type if none was specified
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' since none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
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
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11 --expt-relaxed-constexpr --default-stream per-thread -Wno-deprecated-gpu-targets")

endif()
#############################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-ignored-attributes -Wno-deprecated")

# gather list of source files
if (CORE_USE_CUDA)   
    file(GLOB_RECURSE cuda_files "*.cu")
endif()    

file(GLOB_RECURSE sources "src/*.cc")

list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/parser.cc)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)

# executable

if (CORE_USE_CUDA)
    cuda_add_library(feat_lib SHARED ${sources} ${cuda_files})
else()
    add_library(feat_lib SHARED ${sources})
endif()

# executables

#feat executable
add_executable(feat src/main.cc)

#link libraries for different executables

if (CORE_USE_CUDA)
    target_link_libraries(feat feat_lib shogun ${CUDA_LIBRARIES})
else()
    target_link_libraries(feat feat_lib shogun)
endif()

if (GTEST)
    # Link runTests with what we want to test and the GTest and pthread library
    file(GLOB_RECURSE testsSrc "tests/*.cc")
    
    add_executable(tests ${testsSrc})
    
    if (CORE_USE_CUDA)
        target_link_libraries(tests feat_lib shogun ${CUDA_LIBRARIES} ${GTEST_LIBRARIES} pthread)
    else()
        target_link_libraries(tests feat_lib shogun ${GTEST_LIBRARIES} pthread)
    endif()
endif()


