/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef FEAT_H
#define FEAT_H

//external includes
#include <iostream>
#include <vector>
#include <memory>
#include <shogun/base/init.h>
 
// internal includes
#include "init.h"
#include "util/rnd.h"
#include "util/utils.h"
#include "util/io.h"
#include "params.h"
#include "pop/population.h"
#include "sel/selection.h"
#include "eval/evaluation.h"
#include "vary/variation.h"
#include "model/ml.h"
#include "pop/op/node.h"
#include "pop/archive.h" 

// stuff being used
using Eigen::MatrixXd;
using Eigen::VectorXd;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::vector;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::cout; 

namespace FT{

    using namespace Eval;
    using namespace Vary;
    
    ////////////////////////////////////////////////////////////////////////////////// Declarations
    
    /*!
     * @class Feat
     * @brief main class for the Feat learner.
     *   
     * @details Feat optimizes feature represenations for a given machine learning algorithm. It 
     *			does so by using evolutionary computation to optimize a population of programs. 
     *			Each program represents a set of feature transformations. 
     */
    class Feat 
    {
        public : 
                        
            // Methods 
            
            /// member initializer list constructor
              
            Feat(int pop_size=100, int gens = 100, string ml = "LinearRidgeRegression", 
                   bool classification = false, int verbosity = 2, int max_stall = 0,
                   string sel ="lexicase", string surv="nsga2", float cross_rate = 0.5,
                   char otype='a', string functions = "", 
                   unsigned int max_depth = 3, unsigned int max_dim = 10, int random_state=0, 
                   bool erc = false, string obj="fitness,complexity", bool shuffle=false, 
                   double split=0.75, double fb=0.5, string scorer="", string feature_names="",
                   bool backprop=false,int iters=10, double lr=0.1, int bs=100, int n_threads=0,
                   bool hillclimb=false, string logfile="", int max_time=-1, 
                   bool use_batch = false, bool semantic_xo = false, int print_pop=0);
            
            /// set size of population 
            void set_pop_size(int pop_size);
            
            /// set size of max generations              
            void set_generations(int gens);
                        
            /// set ML algorithm to use              
            void set_ml(string ml);
            
            /// set EProblemType for shogun              
            void set_classification(bool classification);
                 
            /// set level of debug info              
            void set_verbosity(int verbosity);
                        
            /// set maximum stall in learning, in generations
            void set_max_stall(int max_stall);
                        
            /// set selection method              
            void set_selection(string sel);
                        
            /// set survivability              
            void set_survival(string surv);
                        
            /// set cross rate in variation              
            void set_cross_rate(float cross_rate);
                        
            /// set program output type ('f', 'b')              
            void set_otype(char ot);
                        
            /// sets available functions based on comma-separated list.
            void set_functions(string functions);
                        
            /// set max depth of programs              
            void set_max_depth(unsigned int max_depth);
             
            /// set maximum dimensionality of programs              
            void set_max_dim(unsigned int max_dim);
            
            ///set dimensionality as multiple of the number of columns
            void set_max_dim(string str);
            
            /// set seeds for each core's random number generator              
            void set_random_state(int random_state);
                        
            /// flag to set whether to use variable or constants for terminals              
            void set_erc(bool erc);
            
            /// flag to shuffle the input samples for train/test splits
            void set_shuffle(bool sh);

            /// set objectives in feat
            void set_objectives(string obj);
            
            /// set train fraction of dataset
            void set_split(double sp);
            
            ///set data types for input parameters
            void set_dtypes(vector<char> dtypes);

            ///set feedback
            void set_feedback(double fb);

            ///set name for files
            void set_logfile(string s);

            ///set scoring function
            void set_scorer(string s);
            
            void set_feature_names(string s);
            
            void set_feature_names(vector<string>& s);

            /// set constant optimization options
            void set_backprop(bool bp);
            
            void set_hillclimb(bool hc);
            
            void set_iters(int iters);
            
            void set_lr(double lr);
            
            void set_batch_size(int bs);
             
            ///set number of threads
            void set_n_threads(unsigned t);
            
            ///set max time in seconds for fit method
            void set_max_time(int time);
            
            ///set flag to use batch for training
            void set_use_batch();
            
            /// use semantic crossover
            void set_semantic_xo(bool sem_xo=true){params.semantic_xo=sem_xo;};

            void set_print_pop(int pp){ print_pop=pp; };
            /*                                                      
             * getting functions
             */

            ///return population size
            int get_pop_size();
            
            ///return size of max generations
            int get_generations();
            
            ///return ML algorithm string
            string get_ml();
            
            ///return type of classification flag set
            bool get_classification();
            
            ///return maximum stall in learning, in generations
            int get_max_stall();
            
            ///return program output type ('f', 'b')             
            vector<char> get_otypes();
            
            ///return current verbosity level set
            int get_verbosity();
            
            ///return max_depth of programs
            int get_max_depth();
            
            ///return cross rate for variation
            float get_cross_rate();
            
            ///return max size of programs
            int get_max_size();
            
            ///return max dimensionality of programs
            int get_max_dim();
            
            ///return boolean value of erc flag
            bool get_erc();
           
            /// get name
            string get_logfile();

            ///return number of features
            int get_num_features();
            
            ///return whether option to shuffle the data is set or not
            bool get_shuffle();
            
            ///return fraction of data to use for training
            double get_split();
            
            ///add custom node into feat
            /* void add_function(unique_ptr<Node> N){ params.functions.push_back(N->clone()); } */
            
            ///return data types for input parameters
            vector<char> get_dtypes();

            ///return feedback setting
            double get_feedback();
           
            ///return best model
            string get_representation();

            ///return best model: features plus their importances
            string get_model();

            ///get number of parameters in best
            int get_n_params();

            ///get dimensionality of best
            int get_dim();

            ///get dimensionality of best
            int get_complexity();

            ///return population as string
            string get_eqns(bool front=true);
           
            /// return the coefficients or importance scores of the best model. 
            ArrayXd get_coefs();

            /// return the number of nodes in the best model
            int get_n_nodes();

            /// get longitudinal data from file s
            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd>>> get_Z(string s, 
                    long * idx, int idx_size);

            /// destructor             
            ~Feat();
                        
            /// train a model.             
            void fit(MatrixXd& X,
                     VectorXd& y,
                     std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                            std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >());
                            
            void run_generation(unsigned int g,
                            vector<size_t> survivors,
                            DataRef &d,
                            std::ofstream &log,
                            double percentage,
                            unsigned& stall_count);
                     
            /// train a model.             
            void fit(double * X,int rowsX,int colsX, double * Y,int lenY);

            /// train a model, first loading longitudinal samples (Z) from file.
            void fit_with_z(double * X, int rowsX, int colsX, double * Y, int lenY, string s, 
                            long * idx, int idx_size);
           
            /// predict on unseen data.             
            VectorXd predict(MatrixXd& X,
                             std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                                std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >());  
            
            /// predict on unseen data. return CLabels.
            shared_ptr<CLabels> predict_labels(MatrixXd& X,
                             std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                              std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >());  

            /// predict probabilities of each class.
            ArrayXXd predict_proba(MatrixXd& X,
                             std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                              std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >());  
            
            ArrayXXd predict_proba(double * X, int rows_x, int cols_x);
	
            /// predict on unseen data, loading longitudinal samples (Z) from file.
            VectorXd predict_with_z(double * X, int rowsX,int colsX, 
                                    string s, long * idx, int idx_size);

            /// predict probabilities of each class.
            ArrayXXd predict_proba_with_z(double * X, int rowsX,int colsX, 
                                    string s, long * idx, int idx_size);  

            /// predict on unseen data.             
            VectorXd predict(double * X, int rowsX, int colsX);      
            
            /// transform an input matrix using a program.                          
            MatrixXd transform(MatrixXd& X,
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                                 std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >(),
                               Individual *ind = 0);
            
            MatrixXd transform(double * X,  int rows_x, int cols_x);
            
            /// train a model, first loading longitudinal samples (Z) from file.
            MatrixXd transform_with_z(double * X, int rowsX, int colsX, string s, 
                                      long * idx, int idx_size);
            
            /// convenience function calls fit then predict.            
            VectorXd fit_predict(MatrixXd& X,
                                 VectorXd& y,
                                 std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                                 std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >());
                                 
            VectorXd fit_predict(double * X, int rows_x, int cols_x, double * Y, int len_y);
            
            /// convenience function calls fit then transform. 
            MatrixXd fit_transform(MatrixXd& X,
                                   VectorXd& y,
                                   std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                                   std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >());
                                   
            MatrixXd fit_transform(double * X, int rows_x, int cols_x, double * Y, int len_y);
                  
            /// scoring function 
            double score(MatrixXd& X,
                         const VectorXd& y,
                         std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                               std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > >());
            
            /// prints population obj scores each generation 
            void print_population();
        private:
            // Parameters
            Parameters params;    					///< hyperparameters of Feat 
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
            Archive arch;                           ///< pareto front archive
            bool use_arch;                          ///< internal control over use of archive
            string survival;                        ///< stores survival mode
            Normalizer N;                           ///< scales training data.
            string scorer;                          ///< scoring function name.
            // performance tracking
            double best_score;                      ///< current best score
            double best_score_v;                    ///< best validation score
            double best_med_score;                  ///< best median population score
            string str_dim;                         ///< dimensionality as multiple of number of columns 
            void update_best(const DataRef& d, bool val=false);       ///< updates best score   
            void print_stats(std::ofstream& log,
                             double fraction);      ///< prints stats
            Individual best_ind;                    ///< best individual
            string logfile;                         ///< log filename
            int print_pop;                         ///< controls whether pop is printed each gen
            /// method to fit inital ml model            
            void initial_model(DataRef &d);
            /// fits final model to best transformation
            void final_model(DataRef& d);
            /// updates stall count for early stopping
            void update_stall_count(unsigned& stall_count, MatrixXd& F);
    };
}
#endif
