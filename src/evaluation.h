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
    
    class Evaluation 
    {
        // evaluation mixin class for Fewtwo
        public:
        
            Evaluation(){}

            ~Evaluation(){}
                
            // fitness of population.
            void fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                         const Parameters& params);

            // output of an ml model. 
            VectorXd out_ml(MatrixXd& Phi, VectorXd& y, const Parameters& params,
                            std::shared_ptr<ML> ml = nullptr);

            // assign fitness to an individual and to F. 
            void assign_fit(Individual& ind, MatrixXd& F, const VectorXd& yhat, const VectorXd& y,
                            const Parameters& params);       
            

    };
    
    /////////////////////////////////////////////////////////////////////////////////// Definitions  
    
    // fitness of population
    void Evaluation::fitness(Population& pop, const MatrixXd& X, VectorXd& y, MatrixXd& F, 
                 const Parameters& p)
    {
        // Input:
        //      pop: population
        //      X: feature data
        //      y: label
        //      F: matrix of raw fitness values
        //      p: algorithm parameters
        // Output:
        //      F is modified
        //      pop[:].fitness is modified
        
        
        // loop through individuals
        for (auto& ind : pop.individuals)
        {
            // calculate program output matrix Phi
            MatrixXd Phi = ind.out(X, y, p);
            

            // calculate ML model from Phi
            VectorXd yhat = out_ml(Phi,y,p);
            
            // assign F and aggregate fitness
            assign_fit(ind,F,yhat,y,p);
            
            
        }

     
    }
    
    // train ML model and generate output
    VectorXd Evaluation::out_ml(MatrixXd& X, VectorXd& y, const Parameters& params,
                                std::shared_ptr<ML> ml)
    { 
        /* Trains ml on X, y to generate output yhat = f(X). 
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
        std::cout << "X:\n";
        std::cout << X;

        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        auto labels = some<CRegressionLabels>(SGVector<float64_t>(y));
        std::cout << "features and labels defined\n";
        
        std::cout << "loaded features:\n";
        (*features).get_feature_matrix().display_matrix();

        cout << "number of samples:" <<  (*features).get_num_vectors() <<"\n"; 
        cout << "number of features:" << (*features).get_num_features() <<"\n";
        cout << "number of labels:" << (*labels).get_labels().size() << "\n";     

        // preprocess features
        auto Normalize = some<CNormOne>();
        Normalize->init(features);
        auto feat_returned = Normalize->apply_to_feature_matrix(features);
        std::cout << "features normalized\n";
    
        std::cout << "norm features:\n";
        (*features).get_feature_matrix().display_matrix();
        
        feat_returned.display_matrix();

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
        /* assign raw errors to F, and aggregate fitnesses to individuals. 
        *  Input: 
        *       ind: individual 
        *       F: n_samples x pop_size matrix of errors
        *       yhat: predicted output of ind
        *       y: true labels
        *       params: fewtwo parameters
        *  Output:
        *       modifies F and ind.fitness
        */
        
        F.col(ind.loc) = (yhat - y).array().pow(2);

        ind.fitness = F.col(ind.loc).mean();
    }
}
#endif
