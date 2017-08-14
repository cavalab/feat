/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/

//external includes
#include <iostream>
#include <vector>
#include <Eigen/Dense>

// stuff being used
using Eigen::MatrixXd;
using std::vector;
using std::string;

// internal includes
#include "population.h"
#include "selection.h"
#include "evaluation.h"
#include "variation.h"

namespace f2{
    class Fewtwo {
        /* main class for the Fewtwo learner.
        
        Fewtwo optimizes feature represenations for a given 
        machine learning algorithm. It does so by using 
        evolutionary computation to optimize a population of 
        programs. Each program represents a set of feature 
        transformations. 
        */
    public : 
        //////////////////////////////////////////////////////////////////////////////// Parameters
        int pop_size;           // popsize
        int gens;               // number of generations
        string ml;              // machine learner with which Fewtwo is paired 
        //vector<Individual> pop; // population
        MatrixXd F;             // matrix of fitness values for population
        float cross_ratio;      // ratio of crossover to mutation. 
        int max_stall;          // termination criterion if best model does not improve for 
                                // max_stall generations. 0 means not to use
        bool classification;    // whether to conduct classification
        
        // subclasses for main steps of the evolutionary computation routine
        Population pop;         // population of programs
        Selection selection;    // selection algorithm
        Evaluation evaluation;  // evaluation code
        Variation variaton;     // variation operators
        Selection survival;     // survival algorithm

        /////////////////////////////////////////////////////////////////////////////////// Methods 
        Fewtwo(): pop_size(100),
                  gens(100),
                  ml("RidgeRegression"),
                  classification(false);
                  cross_ratio(0.5),
                  max_stall(0),      
                  selection("lexicase"),
                  survival("pareto",true) // true specifies survival verison of selection
      
        {
            // initialization routine.
            // initialize ML object based on ml
            // if classification, default to Logistic Regression
            
        }
        ~Fewtwo(){}
        
        void fit(MatrixXd& X, VectorXd& y){
            // train a model. 
            // steps:
            //  1. fit model yhat = f(X)
            //  2. generate transformations Phi(X) for each individual
            //  3. fit model yhat_new = f( Phi(X)) for each individual
            //  4. evaluate features
            //  5. selection parents
            //  6. produce offspring from parents via variation
            //  7. select surviving individuals from parents and offspring
            
            // initial model on raw input
            initial_model(X,y,estimator);

            // initialize population 
            pop.init(pop_size);

            // resize F to be twice the pop-size x number of samples
            F.resize(2*pop_size,X.rows());
            
            // evaluate initial population
            evaluation.fitness(pop,X,y,F);
            
            vector<size_t> survivors;

            // main generational loop
            for (size_t g = 0; g<gens; ++g)
            {

                // select parents
                vector<size_t> parents = selection.select(F,pop_size);

                // variation to produce offspring
                variation.vary(pop,parents,pop_size);

                // evaluate offspring
                evaluation.fitness(pop,X,y,F);

                // select survivors from combined pool of parents and offspring
                survival.select(F,pop_size);

                // reduce population to survivors
                pop.update(survivors);
            }
        }

        VectorXd predict(MatrixXd& X){
            // predict on unseen data.
            }
        
        MatrixXd transform(MatrixXd& X, Individual ind = Individual()){
            // transform an input matrix using a program.    
        }
        
        VectorXd fit_predict(MatrixXd& X, VectorXd& y){
            // convenience function calls fit then predict.
            fit(X,y);
            return predict(X);
        }
        
        MatrixXd fit_transform(MatrixXd& X, VectorXd& y){
            // convenience function calls fit then transform. 
            fit(X,y);
            return transform(X);
        }
                
        



    };
}
