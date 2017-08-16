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

// stuff being used
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::string;
using std::shared_ptr;
using std::cout; 

// internal includes
#include "params.h"
#include "population.h"
#include "selection.h"
#include "evaluation.h"
#include "variation.h"

namespace FT{
    
    class Fewtwo 
    {
        /* main class for the Fewtwo learner.
        
        Fewtwo optimizes feature represenations for a given machine learning algorithm. It does so
        by using evolutionary computation to optimize a population of programs. Each program 
        represents a set of feature transformations. 
        */
        public : 
            // Parameters
            const Parameters params;    // hyperparameters of Fewtwo 
            MatrixXd F;                 // matrix of fitness values for population
            
            // Methods 
            // member initializer list constructor
            Fewtwo(int pop_size=100, int gens = 100, string ml = "RidgeRegression", 
                   bool classification = false, float cross_ratio = 0.5, int max_stall = 0,
                   string sel ="lexicase", string surv="pareto", char otype='f'): 
                      // construct subclasses
                      params(pop_size, gens, ml, classification, cross_ratio, max_stall, otype),      
                      p_pop(new Population),
                      p_sel(new Selection(sel)),
                      p_surv(new Selection(surv,true)),
                      p_eval(new Evaluation()),
                      p_variation(new Variation())
            {};           

            // destructor
            ~Fewtwo(); 
            
            // train a model.
            void fit(const MatrixXd& X, const VectorXd& y);

            // predict on unseen data.
            VectorXd predict(const MatrixXd& X);
             
            // transform an input matrix using a program.             
            MatrixXd transform(const MatrixXd& X, const Individual ind = Individual());

            // convenience function calls fit then predict.           
            VectorXd fit_predict(const MatrixXd& X, const VectorXd& y)
            { fit(X,y); return predict(X); };
        
            // convenience function calls fit then transform. 
            MatrixXd fit_transform(const MatrixXd& X, const VectorXd& y)
            { fit(X,y); return transform(X); };
                  
        private:
            // subclasses for main steps of the evolutionary computation routine
            shared_ptr<Population> p_pop;       // population of programs
            shared_ptr<Selection> p_sel;        // selection algorithm
            shared_ptr<Evaluation> p_eval;      // evaluation code
            shared_ptr<Variation> p_variation;  // variation operators
            shared_ptr<Selection> p_surv;       // survival algorithm
            //shared_ptr<CMachine> p_est;       // pointer to estimator
            // private methods
            // method to finit inital ml model
            void initial_model(const MatrixXd& X, const VectorXd& y);
    };

    //////////////////////////////////////////////////////////////////////////// Fewtwo Definitions
    
    // train a model
    void Fewtwo::fit(const MatrixXd& X, const VectorXd& y){
        // Parameters:
        //      X: MatrixXd of features
        //      y: VectorXd of labels 
        // Output:
        //      updates best_estimator, hof
        //
        // steps:
        //  1. fit model yhat = f(X)
        //  2. generate transformations Phi(X) for each individual
        //  3. fit model yhat_new = f( Phi(X)) for each individual
        //  4. evaluate features
        //  5. selection parents
        //  6. produce offspring from parents via variation
        //  7. select surviving individuals from parents and offspring
        
        // initial model on raw input
        initial_model(X,y);

        // initialize population 
        p_pop->init(params);

        // resize F to be twice the pop-size x number of samples
        F.resize(int(2*params.pop_size),X.rows());
        
        // evaluate initial population
        p_eval->fitness(*p_pop,X,y,F,params);
        
        vector<size_t> survivors;

        // main generational loop
        for (size_t g = 0; g<params.gens; ++g)
        {

            // select parents
            vector<size_t> parents = p_sel->select(F, params);

            // variation to produce offspring
            p_variation->vary(*p_pop, parents, params);

            // evaluate offspring
            p_eval->fitness(*p_pop, X, y, F, params);

            // select survivors from combined pool of parents and offspring
            survivors = p_surv->select(F, params);

            // reduce population to survivors
            p_pop->update(survivors);
        }
    }
   
}
#endif
