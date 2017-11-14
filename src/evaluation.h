/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef EVALUATION_H
#define EVALUATION_H

// external includes
#include <shogun/machine/Machine.h>
#include <shogun/base/some.h>                                                                       
#include <shogun/base/init.h>                                                                       
#include <shogun/lib/common.h>                                                                      
#include <shogun/labels/RegressionLabels.h>                                                         
#include <shogun/features/Features.h>                                                               
//#include <shogun/preprocessor/PruneVarSubMean.h>        
#include <shogun/preprocessor/NormOne.h>        
#include <cmath>
// internal includes
#include "ml.h"

using namespace shogun;
using Eigen::Map;

// code to evaluate GP programs.
namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Evaluation
     * @brief evaluation mixin class for Fewtwo
     */
    class Evaluation 
    {
        public:
        
            Evaluation(){}

            ~Evaluation(){}
                
            
            /// fitness of population.
            void fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                         const Parameters& params, bool offspring);

            /// output of an ml model. 
            VectorXd out_ml(MatrixXd& Phi, VectorXd& y, const Parameters& params,
                            bool& pass, std::shared_ptr<ML> ml = nullptr);

            /// assign fitness to an individual and to F.  
            void assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, const VectorXd& y,
                            const Parameters& params);       
            

    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions  
    
    // fitness of population
    void Evaluation::fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                 const Parameters& params, bool offspring=false)
    {
    	/*!
         * Input:
         
         *      pop: population
         *      X: feature data
         *      y: label
         *      F: matrix of raw fitness values
         *      p: algorithm parameters
         
         * Output:
         
         *      F is modified
         *      pop[:].fitness is modified
         */
        
        
        unsigned start =0;
        if (offspring) start = F.cols()/2;
        // loop through individuals
        #pragma omp parallel for
        for (unsigned i = start; i<pop.size(); ++i)
        {
            // calculate program output matrix Phi
            params.msg("Generating output for " + pop.individuals[i].get_eqn(), 2);
            MatrixXd Phi = pop.individuals[i].out(X, params, y);            

            // calculate ML model from Phi
            params.msg("ML training on " + pop.individuals[i].get_eqn(), 2);
            bool pass = true;
            VectorXd yhat = out_ml(Phi,y,params,pass);
            if (!pass){
                std::cerr << "Error training eqn " + pop.individuals[i].get_eqn() + "\n";
                std::cerr << "with raw output " << pop.individuals[i].out(X,params,y) << "\n";
                throw;
            }
            // assign F and aggregate fitness
            params.msg("Assigning fitness to " + pop.individuals[i].get_eqn(), 2);
            assign_fit(pop.individuals[i],F,yhat,y,params);
                        
        }

     
    }
    
    // train ML model and generate output
    VectorXd Evaluation::out_ml(MatrixXd& X, VectorXd& y, const Parameters& params,
                                bool& pass, std::shared_ptr<ML> ml)
    { 
    	/*!
         * Trains ml on X, y to generate output yhat = f(X). 
         *
         *  Input: 
         
         *       X: n_features x n_samples matrix
         *       y: n_samples vector of training labels
         *       params: fewtwo parameters
         *       ml: the ML model to be trained on X
         
         *  Output:
         
         *       yhat: n_samples vector of outputs
        */

        if (ml == nullptr)      // make new ML estimator if one is not provided 
            ml = std::make_shared<ML>(params.ml,params.classification);       
        
        
        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " normalize features\n";
        // normalize features
        for (unsigned int i=0; i<X.rows(); ++i){
            if (std::isinf(X.row(i).norm()))
            {
                X.row(i) = VectorXd::Zero(X.row(i).size());
                continue;
            }
            X.row(i) = X.row(i).array() - X.row(i).mean();
            if (X.row(i).norm() > NEAR_ZERO)
                X.row(i).normalize();
        }
        //X.rowwise().normalize();
                // define shogun data
        //if (params.verbosity > 1) 
        //    std::cout << "thread " + std::to_string(omp_get_thread_num()) + " X: " << X << "\n"; 

        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        auto labels = some<CRegressionLabels>(SGVector<float64_t>(y));
         
        // pass data to ml
        ml->p_est->set_labels(labels);

        // train ml
        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " train\n";
        params.msg("ML training on thread" + std::to_string(omp_get_thread_num()) + "...",2," ");
       // #pragma omp critical
        {
            ml->p_est->train(features);
        }
        params.msg("done.",2);
        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " get output\n";
        //get output
        auto y_pred = ml->p_est->apply_regression(features)->get_labels();

        // weights
        vector<double> w = ml->get_weights();

        //std::cout << "thread" + std::to_string(omp_get_thread_num()) + " map to vector\n";

        // map to Eigen vector
        Map<VectorXd> yhat(y_pred.data(),y_pred.size());
        
        if (Eigen::isinf(yhat.array()).any() || Eigen::isnan(yhat.array()).any())
        {
            std::cerr << "inf or nan values in model fit to: " << X << "\n";
            pass = false;
        }
        // return
        return yhat;
    }
    
    // assign fitness to program
    void Evaluation::assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, 
                                const VectorXd& y, const Parameters& params)
    {
        /*!
         * assign raw errors to F, and aggregate fitnesses to individuals. 
         *
         *  Input: 
         *
         *       ind: individual 
         *       F: n_samples x pop_size matrix of errors
         *       yhat: predicted output of ind
         *       y: true labels
         *       params: fewtwo parameters
         *
         *  Output:
         *
         *       modifies F and ind.fitness
        */ 
        assert(F.cols()>ind.loc);
        if (params.classification)  // use classification accuracy
            F.col(ind.loc) = (yhat.cast<int>().array() != y.cast<int>().array()).cast<double>();
        else                        // use mean squared error
            F.col(ind.loc) = (yhat - y).array().pow(2);
        
        // set fitness to average 
        ind.fitness = F.col(ind.loc).mean();
        params.msg("ind " + std::to_string(ind.loc) + " fitnes: " + std::to_string(ind.fitness),2);
    }
}
#endif
