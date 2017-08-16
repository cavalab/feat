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

# autocomplete for YouCompleteMe needs commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# executable
add_executable(fewtwo main.cc)
