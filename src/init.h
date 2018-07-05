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
    #define omp_get_max_threads() 1
    #define omp_set_num_threads( x ) 0
#endif
// stuff being used

#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include <numeric>

using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::vector;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::cout; 
// internal includes

namespace FT{

    static double NEAR_ZERO = 0.0000001;
    static double MAX_DBL = std::numeric_limits<double>::max();
    static double MIN_DBL = std::numeric_limits<double>::lowest();

}

#endif
