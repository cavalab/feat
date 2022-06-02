/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef THRESHOLDOUT_H
#define THRESHOLDOUT_H

//external includes
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <memory>
 
// internal includes
#include "init.h"
#include "util/rnd.h"
#include "util/logger.h"
#include "util/utils.h"
#include "util/io.h"
#include "pop/individual.h"
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
    float certainty; //certainty (beta)
    int B; // budget
    bool gauss;
    float sigma;

    Thresholdout() = default;

    Thresholdout(float t, float beta, int budget=0, bool g=false) 
        : tol(t)
        , certainty(beta)
        , B(budget)
        , gauss(g)
        {
            set_sigma();
        };
    
    void set_sigma()
    {
        float m;
        if (this->B > 0)
            m = B;
        else
            m = 10; // total guess

        sigma = this->tol/(96*log(4*m/this->certainty)); 
    };

    void set_budget(int n){
        set_sigma();

        if ( n < log(6/certainty)/tol )
        {
            cout << "n (" << n << ") < lower bound (" 
                 << log(6/certainty)/tol << endl;
            this->B = std::ceil(std::pow(float(n),2)*std::pow(tol,4)); 
            this->B /= -(log(tol) - log(certainty));
            this->B /= 1600; 
        }
        else
        {
            cout << "B = tol*sigma*n/2 = " 
                 << this->tol << "*" << this->sigma << "*" << n << "/2\n";
            cout << tol << endl;
            cout << tol*sigma << endl;
            cout << tol*sigma*float(n) << endl;
            cout << tol*sigma*float(n)/2 << endl;
            cout << std::ceil(tol*sigma*float(n)/2) << endl;
            auto tmp = std::round(tol*sigma*float(n)/2); 
            cout << "tmp: " << tmp << endl;
            if (tmp > std::numeric_limits<int>::max())
                this->B =  std::numeric_limits<int>::max();
            else
                this->B = int(tmp); 

        }
        cout << "auto-set budget to " << this->B << endl;
    };

    float thresh_validate(const Individual& ind, 
                          const Data& train,
                          const Data& val,
                          bool update_budget=true
    )
    {
        set_sigma();
        // run the thresholdout alg. 
        if (this->B < 1)
        {
            return MAX_FLT;
        }
        float gamma, nu;

        if (this->gauss)
        {
            gamma = r.norm(0,2*sigma);
            nu = r.norm(0,4*sigma);
        }
        else{
            gamma = r.laplace(2*sigma);
            nu = r.laplace(4*sigma);
        }

        float T = 3.0/4.0*this->tol+gamma; 
        float tau = T + nu; 
        int count = 0;
        while ( tau < 0 && count < 10000)
        {
            ++count;
            tau = T + nu;
        }
        if (tau < 0)
            tau = 0.0;
        
        std::ostringstream o;
        o << "this->tol: " << this->tol 
            << ", sigma: " << sigma
            << ", gamma: " << gamma
            << ", nu: " << nu
            << ", T: " << T 
            << ", T + nu: " << tau 
            << ", diff: " << fabs(ind.fitness_v - ind.fitness) << endl;

        logger.log(o.str(),2);
        // one sided overfitting check: if the validation error is much higher
        if (fabs(ind.fitness_v - ind.fitness) >= tau )
        {
            if (update_budget)
            {
                logger.log("model is overfitting! reduced B to "+std::to_string(B),2);
                this->B -= 1;
            }

            float e = this->gauss? r.norm(0,sigma) : r.laplace(sigma); // with noise rate sigma
            int count = 0;
            while ( ind.fitness_v + e < 0 && count < 100)
            {
                ++count;
                e = this->gauss? r.norm(0,sigma) : r.laplace(sigma);
            }
            return ind.fitness_v + e;
        }
        else
            return ind.fitness;
        
    };


};
}

#endif
