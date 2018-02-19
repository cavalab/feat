# version requirement for Eigen
cmake_minimum_required (VERSION 3.0)
# use c++ 14
set (CMAKE_CXX_STANDARD 14)

# project name
project(FeatTests)
# add Eigen include paths
option(EIGEN_DIR "use Eigen environment variable" OFF)
if (EIGEN_DIR)
    include_directories("$ENV{EIGEN3_INCLUDE_DIR}")
else()
    find_package(Eigen3 REQUIRED NO_MODULE)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# add Shogun includes
option(SHOGUN_DIR "use environment Shogun paths" OFF)
if (SHOGUN_DIR)
    include_directories("$ENV{SHOGUN_DIR}")
    link_directories("$ENV{SHOGUN_LIB}")
else()
    find_package(Shogun REQUIRED NO_MODULE)
endif()
 
# Locate GTest
option(GTEST_DIR "use GTest environment path" OFF)
if (GTEST_DIR)
    include_directories("$ENV{GTEST_DIR}/include")

else()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
endif()

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
