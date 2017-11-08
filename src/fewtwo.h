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
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
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
    
    /*!
     * @class Fewtwo
     * @brief main class for the Fewtwo learner.
     *   
     * @details Fewtwo optimizes feature represenations for a given machine learning algorithm. It 
     *			does so by using evolutionary computation to optimize a population of programs. 
     *			Each program represents a set of feature transformations. 
     */
    class Fewtwo 
    {
        public : 
                        
            // Methods 
            
            /// member initializer list constructor
              
            Fewtwo(int pop_size=100, int gens = 100, string ml = "LinearRidgeRegression", 
                   bool classification = false, int verbosity = 1, int max_stall = 0,
                   string sel ="lexicase", string surv="pareto", float cross_rate = 0.5,
                   char otype='a', string functions = "+,-,*,/,exp,log", 
                   unsigned int max_depth = 3, unsigned int max_dim = 10, int random_state=0, 
                   bool erc = false, string obj="fitness,complexity"):
                      // construct subclasses
                      params(pop_size, gens, ml, classification, max_stall, otype, verbosity, 
                             functions, max_depth, max_dim, erc, obj),
                      p_pop( make_shared<Population>(pop_size) ),
                      p_sel( make_shared<Selection>(sel) ),
                      p_surv( make_shared<Selection>(surv, true) ),
                      p_eval( make_shared<Evaluation>() ),
                      p_variation( make_shared<Variation>(cross_rate) ),
                      p_ml( make_shared<ML>(ml, classification) )
            {
                r.set_seed(random_state);
            }
            
            /// set size of population 
            void set_pop_size(int pop_size)
            {
            	params.pop_size = pop_size;
            	p_pop->resize(params.pop_size);
            }            
            
            /// set size of max generations              
            void set_generations(int gens){ params.gens = gens; }         
                        
            /// set ML algorithm to use              
            void set_ml(string ml)
            {
            	params.ml = ml;
            	p_ml = make_shared<ML>(params.ml, params.classification);
            }            
            
            /// set EProblemType for shogun              
            void set_classification(bool classification)
            {
            	params.classification = classification;
            	p_ml = make_shared<ML>(params.ml, params.classification);
            }
                        
            /// set level of debug info              
            void set_verbosity(int verbosity)
            {
            	if(verbosity <=2 && verbosity >=0)
	            	params.verbosity = verbosity;
	            else
	            {
	            	std::cerr << "'" + std::to_string(verbosity) + "' is not a valid verbosity.\n";
	            	std::cerr << "Valid Values :\n\t0 - none\n\t1 - minimal\n\t2 - all\n";
	            }
            }
                        
            /// set maximum stall in learning, in generations
            void set_max_stall(int max_stall){	params.max_stall = max_stall; }
                        
            /// set selection method              
            void set_selection(string sel){ p_sel = make_shared<Selection>(sel); }
                        
            /// set survivability              
            void set_survival(string surv){ p_surv = make_shared<Selection>(surv, true); }
                        
            /// set cross rate in variation              
            void set_cross_rate(float cross_rate){	p_variation->set_cross_rate(cross_rate); }
                        
            /// set program output type ('f', 'b')              
            void set_otype(char o_type){ params.otypes.clear(); params.otypes.push_back(o_type); }
                        
            /// sets available functions based on comma-separated list.
            void set_functions(string functions){ params.set_functions(functions); }
                        
            /// set max depth of programs              
            void set_max_depth(unsigned int max_depth){ params.set_max_depth(max_depth); }
            
            /// set maximum dimensionality of programs              
            void set_max_dim(unsigned int max_dim){	params.set_max_dim(max_dim); }
                        
            /// set seeds for each core's random number generator              
            void set_random_state(int random_state){ r.set_seed(random_state); }
                        
            /// flag to set whether to use variable or constants for terminals              
            void set_erc(bool erc){ params.erc = erc; }
                        
            /// destructor             
            ~Fewtwo(){} 
                        
            /// train a model.             
            void fit(MatrixXd& X, VectorXd& y);
            
            /// predict on unseen data.             
            VectorXd predict(MatrixXd& X);        
            
            /// transform an input matrix using a program.                          
            MatrixXd transform(const MatrixXd& X,  Individual *ind = 0);
            
            /// convenience function calls fit then predict.            
            VectorXd fit_predict(MatrixXd& X, VectorXd& y){ fit(X,y); return predict(X); }
        
            
            /// convenience function calls fit then transform. 
            MatrixXd fit_transform(MatrixXd& X, VectorXd& y){ fit(X,y); return transform(X); }
                  
        private:
            // Parameters
            Parameters params;    					///< hyperparameters of Fewtwo 
            MatrixXd F;                 			///< matrix of fitness values for population
            Timer timer;                            ///< start time of training
            // subclasses for main steps of the evolutionary computation routine
            shared_ptr<Population> p_pop;       	///< population of programs
            shared_ptr<Selection> p_sel;        	///< selection algorithm
            shared_ptr<Evaluation> p_eval;      	///< evaluation code
            shared_ptr<Variation> p_variation;  	///< variation operators
            shared_ptr<Selection> p_surv;       	///< survival algorithm
            shared_ptr<ML> p_ml;                	///< pointer to machine learning class
            // performance tracking
            double best_score;                      ///< current best score 
            void update_best();                     ///< updates best score   
            void print_stats(unsigned int);        ///< prints stats
            Individual best_ind;                                            ///< best individual
            /// method to fit inital ml model            
            void initial_model(MatrixXd& X, VectorXd& y);
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    void Fewtwo::fit(MatrixXd& X, VectorXd& y)
    {
        /*!
         *  Input:
         
         *       X: n_features x n_samples MatrixXd of features
         *       y: VectorXd of labels 
         
         *  Output:
         
         *       updates best_estimator, hof
        
         *   steps:
         *	   1. fit model yhat = f(X)
         *	   2. generate transformations Phi(X) for each individual
         *	   3. fit model yhat_new = f( Phi(X)) for each individual
         *	   4. evaluate features
         *	   5. selection parents
         *	   6. produce offspring from parents via variation
         *	   7. select surviving individuals from parents and offspring
         */
        // start the clock
        timer.Reset();

        // define terminals based on size of X
        params.set_terminals(X.rows()); 
        
        // initial model on raw input
        params.msg("Fitting initial model", 1);
        initial_model(X,y);
                      
        // initialize population 
        params.msg("Initializing population", 1);
        p_pop->init(best_ind,params);
        params.msg("Initial population:\n"+p_pop->print_eqns(","),2);

        // resize F to be twice the pop-size x number of samples
        F.resize(X.cols(),int(2*params.pop_size));
        
        // evaluate initial population
        params.msg("Evaluating initial population",1);
        p_eval->fitness(*p_pop,X,y,F,params);
        
        vector<size_t> survivors;

        // main generational loop
        for (unsigned int g = 0; g<params.gens; ++g)
        {

            // select parents
            params.msg("selection..", 2);
            vector<size_t> parents = p_sel->select(*p_pop, F, params);
            params.msg("parents:\n"+p_pop->print_eqns(","), 2);          
            
            // variation to produce offspring
            params.msg("variation...", 2);
            p_variation->vary(*p_pop, parents, params);
            params.msg("offspring:\n" + p_pop->print_eqns(true), 2);

            // evaluate offspring
            params.msg("evaluating offspring...", 2);
            p_eval->fitness(*p_pop, X, y, F, params, true);

            // select survivors from combined pool of parents and offspring
            params.msg("survival", 2);
            survivors = p_surv->survive(*p_pop, F, params);
           
            // reduce population to survivors
            params.msg("shrinking pop to survivors...",2);
            p_pop->update(survivors);
            params.msg("survivors:\n" + p_pop->print_eqns(), 2);

            update_best();
            if (params.verbosity>0) print_stats(g+1);
        }
        params.msg("finished",1);
        params.msg("best representation: " + best_ind.get_eqn(),1);
        params.msg("score: " + std::to_string(best_score), 1);
    }

    void Fewtwo::initial_model(MatrixXd& X, VectorXd& y)
    {
        /*!
         * fits an ML model to the raw data as a starting point.
         */
        bool pass = true;
        VectorXd yhat = p_eval->out_ml(X,y,params,pass,p_ml);

        // set terminal weights based on model
        params.set_term_weights(p_ml->get_weights());

        // assign best score as MSE
        best_score = (yhat-y).array().pow(2).mean();

        // initialize best_ind to be all the features
        best_ind = Individual();
        for (unsigned i =0; i<X.rows(); ++i)
            best_ind.program.push_back(params.terminals[i]);
        best_ind.fitness = best_score;
    }

    MatrixXd Fewtwo::transform(const MatrixXd& X, Individual *ind)
    {
        /*!
         * Transforms input data according to ind or best ind, if ind is undefined.
         */
        if (ind == 0)        // if ind is empty, predict with best_ind
        {
            if (best_ind.program.size()==0){
                std::cerr << "You need to train a model using fit() before making predictions.\n";
                throw;
            }
            return best_ind.out(X,params);
        }
        return ind->out(X,params);
    }
    
    VectorXd Fewtwo::predict(MatrixXd& X)
    {
        MatrixXd Phi = transform(X);
        auto PhiSG = some<CDenseFeatures<float64_t>>(SGMatrix<float64_t>(Phi));
        auto y_pred = p_ml->p_est->apply_regression(PhiSG)->get_labels();
        return Eigen::Map<VectorXd>(y_pred.data(),y_pred.size());
    }

    void Fewtwo::update_best()
    {
        for (const auto& i: p_pop->individuals)
        {
            if (i.fitness < best_score)
            {
                best_score = i.fitness;
                best_ind = i;
            }
        }
 
    }
 
    void Fewtwo::print_stats(unsigned int g)
    {
        vector<size_t> pf = p_pop->sorted_front();
        double med_score = median(F.colwise().mean().array());
        string bar, space = "";
        for (unsigned int i = 0; i<50; ++i){
            if (i <= 50*g/params.gens) bar += "/";
            else space += " ";
        }
        std::cout.precision(3);
        std::cout << std::scientific;
        std::cout << "Generation " << g << "/" << params.gens << " [" + bar + space + "]\n";
        std::cout << "Min Loss\tMedian Loss\tTime (s)\n"
                  <<  best_score << "\t" << med_score << "\t" << timer << "\n";
        std::cout << "Representation Pareto Front--------------------------------------\n";
        std::cout << "Rank\tComplexity\tLoss\tRepresentation\n";
       // for (const auto& i : pf){
       //     std::cout << p_pop->individuals[i].complexity() << "\t" << (*p_pop)[i].fitness 
       //               << "\t" << p_pop->individuals[i].get_eqn() << "\n";
       // }
        unsigned j =0;
        unsigned n = 1;
        while (j<std::min(10,params.pop_size)){            
			vector<size_t> f = p_pop->sorted_front(n);
            j+= f.size();
            ++n;
			for (const auto& i : f){
            	std::cout << p_pop->individuals[i].rank << "\t" << 
                    p_pop->individuals[i].complexity() << "\t" << (*p_pop)[i].fitness 
                      << "\t" << p_pop->individuals[i].get_eqn() << "\n";
            }
        }
        std::cout << "\n\n";
        
    }
}
#endif
