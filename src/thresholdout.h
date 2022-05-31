/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef THRESHOLDOUT_H
#define THRESHOLDOUT_H

//external includes
#include <iostream>
#include <vector>
#include <memory>
 
// internal includes
#include "init.h"
#include "util/rnd.h"
#include "util/logger.h"
#include "util/utils.h"
#include "util/io.h"
// #include "params.h"
#include "pop/individual.h"
// #include "sel/selection.h"
// #include "eval/evaluation.h"
// #include "vary/variation.h"
// #include "model/ml.h"
// #include "pop/op/node.h"
// #include "pop/archive.h" 
// #include "pop/op/longitudinal/n_median.h"

// // stuff being used
// using Eigen::MatrixXf;
// using Eigen::VectorXf;
// typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::vector;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::cout; 
using std::log;
using std::pow;

/**
* @namespace FT
* @brief main Feat namespace
*/
namespace FT{

struct Thresholdout {
    // implements thresholdout alg from Dwork et al 2015. 
    float tol; // tolerance (tau)
    float sigma; // noise rate (sigma)
    float certainty; //certainty (beta)
    int B; // budget
    bool gauss;

    Thresholdout() = default;

    Thresholdout(float t, float beta, int budget, bool g=false) 
        : tol(t)
        , sigma(2*t)
        , certainty(beta)
        , B(budget)
        , gauss(g)
        {};

    void set_budget(int n){
        if ( n < log(6/certainty)/tol )
        {
            this->B = pow(n,2)*pow(tol,4); 
            this->B /= -(log(tol) - log(certainty));
            this->B /= 1600; 
        }
        else
        {
            this->B = pow(this->tol,2)*n;

        }
        cout << "auto-set budget to " << B << endl;
    };

    float thresh_validate(const Individual& ind, 
                          const Data& train,
                          const Data& val
    )
    {
        
        // run the thresholdout alg. 
        if (this->B < 1)
        {
            cout << "budget exceeded!\n";
            return MAX_FLT;
        }
        float gamma = this->gauss? r.norm(0,4*sigma) : r.laplace(4*sigma); // with variation tol
        float tau = this->tol + gamma;
        // one sided overfitting check: if the validation error is much higher
        if (fabs(ind.fitness_v - ind.fitness) >= tau )
        {
            logger.log("model is overfitting! reduced B to "+std::to_string(B),2);
            --this->B;
            float e = this->gauss? r.norm(0,sigma) : r.laplace(sigma); // with noise rate sigma
            this->tol += this->gauss? r.norm(0,2*sigma): r.laplace(2*sigma);
            return ind.fitness_v + e; 
        }
        else
            return ind.fitness;
        
    };


};
}

#endif
