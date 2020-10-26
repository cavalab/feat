/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef INIT_H
#define INIT_H

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_num_threads() 1
    #define omp_get_max_threads() 1
    #define omp_set_num_threads( x ) 0
#endif
// stuff being used

#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include <numeric>
#include <map>
using Eigen::MatrixXf;
using Eigen::VectorXf;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
typedef Eigen::Matrix<bool,Eigen::Dynamic,1> VectorXb;
typedef Eigen::Matrix<long,Eigen::Dynamic,1> VectorXl;
using std::vector;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::cout; 
typedef std::map<string, 
                 std::pair<vector<Eigen::ArrayXf>, vector<Eigen::ArrayXf>>
                > LongData;
// internal includes
#include "util/json.hpp"
using nlohmann::json;

namespace FT{

    static float NEAR_ZERO = 0.0000001;
    static float MAX_FLT = std::numeric_limits<float>::max();
    static float MIN_FLT = std::numeric_limits<float>::lowest();

}

#endif
