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
    
    //////////////////////////////////////////////////////////////////////////////// Declarations
    
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
        private:
            
            

    };
    
    /////////////////////////////////////////////////////////////////////////////// Definitions  
    
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
        if (ml == nullptr)      // make new ML estimator if one is not provided 
        {
            ml = std::make_shared<ML>(params.ml);
        }
               

        // define shogun data
        
        X.transposeInPlace();

        auto features = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(X));
        auto labels = some<CRegressionLabels>(SGVector<float64_t>(y));
        std::cout << "features and labels defined\n";

        cout << "number of samples:" <<  (*features).get_num_vectors() <<"\n"; 
        cout << "number of features:" << (*features).get_num_features() <<"\n";
        cout << "number of labels:" << (*labels).get_labels().size() << "\n";     

        // preprocess features
        auto Normalize = some<CNormOne>();
        Normalize->init(features);
        Normalize->apply_to_feature_matrix(features);
        std::cout << "features normalized\n";

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
    }
}
#endif
