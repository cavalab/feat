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
                   bool erc = false, string obj="fitness,complexity",bool shuffle=false, 
                   double split=0.75):
                      // construct subclasses
                      params(pop_size, gens, ml, classification, max_stall, otype, verbosity, 
                             functions, max_depth, max_dim, erc, obj, shuffle, split),
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
            void set_verbosity(int verbosity){ params.set_verbosity(verbosity); }
                        
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
            
            ///return population size
            int get_pop_size(){ return params.pop_size; }
            
            ///return size of max generations
            int get_generations(){ return params.gens; }
            
            ///return ML algorithm string
            string get_ml(){ return params.ml; }
            
            ///return type of classification flag set
            bool get_classification(){ return params.classification; }
            
            ///return maximum stall in learning, in generations
            int get_max_stall() { return params.max_stall; }
            
            ///return program output type ('f', 'b')             
            vector<char> get_otypes(){ return params.otypes; }
            
            ///return current verbosity level set
            int get_verbosity(){ return params.verbosity; }
            
            ///return max_depth of programs
            int get_max_depth(){ return params.max_depth; }
            
            ///return max size of programs
            int get_max_size(){ return params.max_size; }
            
            ///return max dimensionality of programs
            int get_max_dim(){ return params.max_dim; }
            
            ///return boolean value of erc flag
            bool get_erc(){ return params.erc; }
            
            ///return number of features
            int get_num_features(){ return params.num_features; }
                        
            /// flag to shuffle the input samples for train/test splits
            void set_shuffle(bool sh){params.shuffle = sh;}

            /// set train fraction of dataset
            void set_split(double sp){params.split = sp;}

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
            MatrixXd F_v;                           ///< matrix of validation scores
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
            void print_stats(unsigned int);         ///< prints stats
            Individual best_ind;                    ///< best individual
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

        // split data into training and test sets
        MatrixXd X_t(X.rows(),int(X.cols()*params.split));
        MatrixXd X_v(X.rows(),int(X.cols()*(1-params.split)));
        VectorXd y_t(int(y.size()*params.split)), y_v(int(y.size()*(1-params.split)));
        train_test_split(X,y,X_t,X_v,y_t,y_v,params.shuffle);
        
        // define terminals based on size of X
        params.set_terminals(X.rows()); 
        
        // initial model on raw input
        params.msg("Fitting initial model", 1);
        initial_model(X,y);
        params.msg("Initial score: " + std::to_string(best_score), 1);

        // initialize population 
        params.msg("Initializing population", 1);
        p_pop->init(best_ind,params);
        params.msg("Initial population:\n"+p_pop->print_eqns(","),2);

        // resize F to be twice the pop-size x number of samples
        F.resize(X_t.cols(),int(2*params.pop_size));
       
        // evaluate initial population
        params.msg("Evaluating initial population",1);
        p_eval->fitness(*p_pop,X_t,y_t,F,params);
        
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
            p_eval->fitness(*p_pop, X_t, y_t, F, params, true);

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
        params.msg("best training representation: " + best_ind.get_eqn(),1);
        params.msg("train score: " + std::to_string(best_score), 1);
        // evaluate population on validation set
        F_v.resize(X_v.cols(),int(2*params.pop_size)); 
        p_eval->fitness(*p_pop, X_v, y_v, F_v, params);
        initial_model(X_v, y_v);        // calculate baseline model validation score
        update_best();                  // get the best validation model
        params.msg("best validation representation: " + best_ind.get_eqn(),1);
        params.msg("validation score: " + std::to_string(best_score), 1);
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

        if (params.classification)  // assign best score as mean accuracy
            best_score = (yhat.cast<int>().array() != y.cast<int>().array()).cast<double>().mean();
        else                        // assign best score as MSE
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
        unsigned num_models = std::min(20,p_pop->size());
        double med_score = median(F.colwise().mean().array());  // median loss
        ArrayXd Sizes(p_pop->size()); unsigned i = 0;           // collect program sizes
        for (const auto& p : p_pop->individuals){ Sizes(i) = p.size(); ++i;}
        double med_size = median(Sizes);                        // median program size
        string bar, space = "";                                 // progress bar
        for (unsigned int i = 0; i<50; ++i){
            if (i <= 50*g/params.gens) bar += "/";
            else space += " ";
        }
        std::cout.precision(3);
        std::cout << std::scientific;
        std::cout << "Generation " << g << "/" << params.gens << " [" + bar + space + "]\n";
        std::cout << "Min Loss\tMedian Loss\tMedian Program Size\tTime (s)\n"
                  <<  best_score << "\t" << med_score << "\t" << med_size << "\t" << timer << "\n";
        std::cout << "Representation Pareto Front--------------------------------------\n";
        std::cout << "Rank\tComplexity\tLoss\tRepresentation\n";
       
        // printing 10 individuals from the pareto front
        unsigned n = 1;
        vector<size_t> f = p_pop->sorted_front(n);
        vector<size_t> fnew(2,0);
        while (f.size() < num_models && fnew.size()>1)
        {
            fnew = p_pop->sorted_front(++n);                
            f.insert(f.end(),fnew.begin(),fnew.end());
        }
        
        for (unsigned j = 0; j < std::min(num_models,unsigned(f.size())); ++j)
        {          
            std::cout << p_pop->individuals[f[j]].rank << "\t" 
                      <<  p_pop->individuals[f[j]].complexity() << "\t" << (*p_pop)[f[j]].fitness 
                      << "\t" << p_pop->individuals[f[j]].get_eqn() << "\n";  
        }
        std::cout << "\n\n";
        
    }
}
#endif
