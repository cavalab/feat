/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef FEWTWO_H
#define FEWTWO_H

//external includes
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <memory>
#include <shogun/base/init.h>
#include <omp.h>

// stuff being used
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;
using std::cout; 

// internal includes
#include "rnd.h"
#include "utils.h"
#include "params.h"
#include "population.h"
#include "selection.h"
#include "evaluation.h"
#include "variation.h"
#include "ml.h"

namespace FT{
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    
    class Fewtwo 
    {
        /* main class for the Fewtwo learner.
        
        Fewtwo optimizes feature represenations for a given machine learning algorithm. It does so
        by using evolutionary computation to optimize a population of programs. Each program 
        represents a set of feature transformations. 
        */
        public : 
                        
            // Methods 
            // member initializer list constructor
            Fewtwo(int pop_size=100, int gens = 100, string ml = "LinearRidgeRegression", 
                   bool classification = false, int verbosity = 1, int max_stall = 0,
                   string sel ="lexicase", string surv="pareto", float cross_rate = 0.5,
                   char otype='f', string functions = "+,-,*,/,exp,log", 
                    unsigned int max_depth = 3, unsigned int max_dim = 10, int random_state=0, 
                    vector<double> term_weights = vector<double>()):
                      // construct subclasses
                      params(pop_size, gens, ml, classification, max_stall, otype, verbosity, 
                             functions, term_weights, max_depth, max_dim),
                      p_pop( make_shared<Population>(pop_size) ),
                      p_sel( make_shared<Selection>(sel) ),
                      p_surv( make_shared<Selection>(surv, true) ),
                      p_eval( make_shared<Evaluation>() ),
                      p_variation( make_shared<Variation>(cross_rate) ),
                      p_ml( make_shared<ML>(ml, classification) )
            {
                r.set_seed(random_state);                    
            }           

            // destructor
            ~Fewtwo(){} 
            
            // train a model.
            void fit(MatrixXd& X, VectorXd& y);

            // predict on unseen data.
            VectorXd predict(const MatrixXd& X);
             
            // transform an input matrix using a program.             
            MatrixXd transform(const MatrixXd& X, const Individual ind = Individual());

            // convenience function calls fit then predict.           
            VectorXd fit_predict(MatrixXd& X, VectorXd& y)
            { fit(X,y); return predict(X); };
        
            // convenience function calls fit then transform. 
            MatrixXd fit_transform(MatrixXd& X, VectorXd& y)
            { fit(X,y); return transform(X); };
                  
        private:
            // Parameters
            Parameters params;    // hyperparameters of Fewtwo 
            MatrixXd F;                 // matrix of fitness values for population
            
            // subclasses for main steps of the evolutionary computation routine
            shared_ptr<Population> p_pop;       // population of programs
            shared_ptr<Selection> p_sel;        // selection algorithm
            shared_ptr<Evaluation> p_eval;      // evaluation code
            shared_ptr<Variation> p_variation;  // variation operators
            shared_ptr<Selection> p_surv;       // survival algorithm
            shared_ptr<ML> p_ml;                // pointer to machine learning class
            // private methods
            // method to finit inital ml model
            void initial_model(MatrixXd& X, VectorXd& y);
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    void Fewtwo::fit(MatrixXd& X, VectorXd& y){
        /*  trains a fewtwo model. 

            Parameters:
                X: n_features x n_samples MatrixXd of features
                y: VectorXd of labels 
            Output:
                updates best_estimator, hof
        
            steps:
            1. fit model yhat = f(X)
            2. generate transformations Phi(X) for each individual
            3. fit model yhat_new = f( Phi(X)) for each individual
            4. evaluate features
            5. selection parents
            6. produce offspring from parents via variation
            7. select surviving individuals from parents and offspring
        */
        
        // define terminals based on size of X
        params.set_terminals(X.rows()); 

        // initial model on raw input
        params.msg("Fitting initial model", 1);
        initial_model(X,y);
        
                
        // initialize population 
        params.msg("Initializing population", 1);
        p_pop->init(params);

        // resize F to be twice the pop-size x number of samples
        F.resize(int(2*params.pop_size),X.cols());
        
        // evaluate initial population
        params.msg("Evaluating initial population",1);
        p_eval->fitness(*p_pop,X,y,F,params);
        
        vector<size_t> survivors;

        // main generational loop
        for (size_t g = 0; g<params.gens; ++g)
        {
            params.msg("g " + std::to_string(g),1);

            // select parents
            params.msg("selection", 1);
            vector<size_t> parents = p_sel->select(F, params);

            // variation to produce offspring
            params.msg("variation", 1);
            p_variation->vary(*p_pop, parents, params);

            // evaluate offspring
            params.msg("evaluation", 1);
            p_eval->fitness(*p_pop, X, y, F, params);

            // select survivors from combined pool of parents and offspring
            params.msg("survival", 1);
            survivors = p_surv->select(F, params);

            // reduce population to survivors
            p_pop->update(survivors);
        }
    }

    void Fewtwo::initial_model(MatrixXd& X, VectorXd& y)
    {
        /* fits an ML model to the raw data as a starting point. */
         
        VectorXd yhat = p_eval->out_ml(X,y,params,p_ml);

        // set terminal weights based on model
        params.set_term_weights(p_ml->get_weights());
    }

   
}
#endif
