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
                
            /*!
             * @brief fitness of population.
             */
            void fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                         const Parameters& params);

            /*!
             * @brief output of an ml model. 
             */
            VectorXd out_ml(MatrixXd& Phi, VectorXd& y, const Parameters& params,
                            std::shared_ptr<ML> ml = nullptr);

            /*! 
             * @brief assign fitness to an individual and to F. 
             */
            void assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, const VectorXd& y,
                            const Parameters& params);       
            

    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions  
    
    // fitness of population
    void Evaluation::fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                 const Parameters& params)
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
        
        char otype = params.otype;        
        // loop through individuals
        for (auto& ind : pop.individuals)
        {
            // calculate program output matrix Phi
            params.msg("Generating output for " + ind.get_eqn(otype), 0);
            MatrixXd Phi = ind.out(X, y, params);
            

            // calculate ML model from Phi
            params.msg("ML training on " + ind.get_eqn(otype), 0);
            VectorXd yhat = out_ml(Phi,y,params);
            
            // assign F and aggregate fitness
            params.msg("Assigning fitness to " + ind.get_eqn(otype), 0);
            assign_fit(ind,F,yhat,y,params);
            
            
        }

     
    }
    
    // train ML model and generate output
    VectorXd Evaluation::out_ml(MatrixXd& X, VectorXd& y, const Parameters& params,
                                std::shared_ptr<ML> ml)
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
        {
            ml = std::make_shared<ML>(params.ml,params.classification);
        }
        
        
        // define shogun data
        
        //X.transposeInPlace();
        std::cout << "Phi:\n";
        std::cout << X << "\n";

        // normalize features
        for (unsigned int i=0; i<X.rows(); ++i){
            X.row(i) = X.row(i).array() - X.row(i).mean();
            if (X.row(i).norm() > NEAR_ZERO)
                X.row(i).normalize();
        }
        std::cout << "normalized Phi:\n";
        std::cout << X << "\n";

        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        auto labels = some<CRegressionLabels>(SGVector<float64_t>(y));
        std::cout << "features and labels defined\n";
        
        //std::cout << "loaded features:\n";
        //(*features).get_feature_matrix().display_matrix();

        cout << "number of samples:" <<  (*features).get_num_vectors() <<"\n"; 
        cout << "number of features:" << (*features).get_num_features() <<"\n";
        cout << "number of labels:" << (*labels).get_labels().size() << "\n";   
       
    
        std::cout << "labels:\n";
        (*labels).get_labels().display_vector();

        // pass data to ml
        //ml->p_est->set_features(features);
        ml->p_est->set_labels(labels);
        std::cout << "ml labels set\n";

        // train ml
        ml->p_est->train(features);
        std::cout << "ml model trained\n";

        //get output
        auto y_pred = ml->p_est->apply_regression(features)->get_labels();
        std::cout << "prediction generated\n";

        // weights
        vector<double> w = ml->get_weights();
        std::cout << "weights: ";
        for (int i =0; i < w.size(); ++i) std::cout << w[i] << ", ";
        std::cout << "\n";

        // map to Eigen vector
        Map<VectorXd> yhat(y_pred.data(),y_pred.size());
        std::cout << "mapped to eigen matrix\n";
        
        // exit shogun
        //exit_shogun();

        cout << "y true: ";
        for (size_t i = 0; i<y.size(); ++i) cout << y(i) << " ";
        cout << "\n";

        cout << "y_pred: ";
        for (size_t i = 0; i<y_pred.size(); ++i) cout << y_pred[i] << " ";
        cout << "\n";

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
         
         *       modifies F and ind.fitness
        */
        std::cout << "F: " << F.rows() << " x " << F.cols() << "\n";
        std::cout << "ind.loc: " << ind.loc << "\n";
        std::cout << "yhat " << yhat.size() << "\n";
        std::cout << "y: " << y.size() << "\n";

        F.col(ind.loc) = (yhat - y).array().pow(2);
        
        ind.fitness = F.col(ind.loc).mean();
        params.msg("ind " + std::to_string(ind.loc) + " fitnes: " + std::to_string(ind.fitness),0);
    }
}
#endif
