# version requirement for Eigen
cmake_minimum_required (VERSION 3.0)
# use c++ 11
set (CMAKE_CXX_STANDARD 11)

# project name
project(FewtwoTests)

# add Eigen include paths
find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
 
# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# add Shogun includes
find_package(Shogun REQUIRED NO_MODULE)
# find Shogun library
#find_library(Shogun REQUIRED NO_MODULE)

# add OpenMP support
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# autocomplete for YouCompleteMe needs commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
 
# Link runTests with what we want to test and the GTest and pthread library
add_executable(runTests gtest.cc)
target_link_libraries(runTests shogun ${GTEST_LIBRARIES} pthread)
