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
#include "util/logger.h"
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
#include "pop/op/longitudinal/n_median.h"

#ifdef USE_CUDA
    #include "pop/cuda-op/cuda_utils.h" 
    #define GPU true
#else
    #define GPU false
    #define initialize_cuda() 0
#endif

// stuff being used
using Eigen::MatrixXf;
using Eigen::VectorXf;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::vector;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::cout; 

/**
* @namespace FT
* @brief main Feat namespace
*/
namespace FT{

using namespace Eval;
using namespace Vary;

////////////////////////////////////////////////////////////////// Declarations

/*!
 * @class Feat
 * @brief main class for the Feat learner.
 *   
 * @details Feat optimizes feature represenations for a given machine 
 * learning algorithm. It does so by using evolutionary computation to 
 * optimize a population of programs. Each program represents a set of 
 * feature transformations. 
 */
class Feat 
{
    public : 
                    
        // Methods 
        
        /// member initializer list constructor
          
        Feat(int pop_size=100, int gens = 100, 
             string ml = "LinearRidgeRegression", 
             bool classification = false, int verbosity = 2, 
             int max_stall = 0, string sel ="lexicase", 
             string surv="nsga2", float cross_rate = 0.5, 
             float root_xo_rate = 0.5, char otype='a', 
             string functions = "", unsigned int max_depth = 3, 
             unsigned int max_dim = 10, int random_state=-1, 
             bool erc = false, string obj="fitness,complexity", 
             bool shuffle=true, float split=0.75, float fb=0.5, 
             string scorer="", string feature_names="",
             bool backprop=false,int iters=10, float lr=0.1, 
             int batch_size=0, int n_jobs=0, bool hillclimb=false, 
             string logfile="", int max_time=-1, bool residual_xo = false, 
             bool stagewise_xo = false, bool stagewise_tol = true, 
             bool softmax_norm=false, int save_pop=0, bool normalize=true, 
             bool val_from_arch=true, bool corr_delete_mutate=false, 
             float simplify=0.0, string protected_groups="",
             bool tune_initial=false, bool tune_final=true,
             string starting_pop="");
        
        /// set size of population 
        void set_pop_size(int pop_size);
        
        /// set size of max generations              
        void set_gens(int gens);
                    
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
        
        /// set root xo rate in variation              
        void set_root_xo_rate(float cross_rate);
        float get_root_xo_rate(){return this->params.root_xo_rate;};
                    
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
        int get_random_state() { return this->random_state; };
        /// returns the actual seed determined by the input argument.
        int get_random_state_() { return r.get_seed(); };
                    
        /// flag to set whether to use variable or constants for terminals
        void set_erc(bool erc);
        
        /// flag to shuffle the input samples for train/test splits
        void set_shuffle(bool sh);

        /// set objectives in feat
        void set_objectives(string obj);
        
        /// set train fraction of dataset
        void set_split(float sp);
        
        ///set data types for input parameters
        void set_dtypes(vector<char> dtypes);

        ///set feedback
        void set_fb(float fb);

        ///set name for files
        void set_logfile(string s);

        ///set scoring function
        void set_scorer(string s);
        // returns the input argument for scorer.
        string get_scorer();
        // returns the actual scorer determined by the input argument.
        string get_scorer_();
        
        void set_feature_names(string s){params.set_feature_names(s); };
        string get_feature_names(){return params.get_feature_names(); };

        /// set constant optimization options
        void set_backprop(bool bp);
        bool get_backprop(){return params.backprop;};
        
        void set_simplify(float s);
        float get_simplify(){return simplify;};

        void set_corr_delete_mutate(bool s);
        bool get_corr_delete_mutate(){return params.corr_delete_mutate;};
        
        void set_hillclimb(bool hc);
        bool get_hillclimb(){return params.hillclimb;};
        
        void set_iters(int iters);
        int get_iters(){return params.bp.iters;};
        
        void set_lr(float lr);
        float get_lr(){return params.bp.learning_rate;};
        
        int get_batch_size(){return params.bp.batch_size;};
        void set_batch_size(int bs);
         
        ///set number of threads
        void set_n_jobs(unsigned t);
        int get_n_jobs(){return omp_get_num_threads();};
        
        ///set max time in seconds for fit method
        void set_max_time(int time);
        int get_max_time(){return params.max_time;};
        
        ///set flag to use batch for training
        void set_use_batch();
        
        /// use residual crossover
        void set_residual_xo(bool res_xo=true){params.residual_xo=res_xo;};
        bool get_residual_xo(){return params.residual_xo;};
        
        /// use stagewise crossover
        void set_stagewise_xo(bool sem_xo=true){ params.stagewise_xo=sem_xo; };
        bool get_stagewise_xo(){return params.stagewise_xo;};

        void set_stagewise_xo_tol(int tol){ params.stagewise_xo_tol = tol;};
        int get_stagewise_xo_tol(){return params.stagewise_xo_tol;};
        
        /// use softmax
        void set_softmax_norm(bool sftmx=true){params.softmax_norm=sftmx;};
        bool get_softmax_norm(){return params.softmax_norm;};

        void set_save_pop(int pp){ save_pop=pp; };
        int get_save_pop(){ return save_pop; };
        
        void set_starting_pop(string sp){ starting_pop=sp; };
        string get_starting_pop(){ return starting_pop; };

        void set_normalize(bool in){params.normalize = in;};
        bool get_normalize(){return params.normalize;};


        /*                                                      
         * getting functions
         */

        ///return population size
        int get_pop_size();
        
        ///return archive size
        int get_archive_size(){ return this->archive.individuals.size(); };

        ///return size of max generations
        int get_gens();
        
        ///return ML algorithm string
        string get_ml();
        
        ///return type of classification flag set
        bool get_classification();
        
        ///return maximum stall in learning, in generations
        int get_max_stall();
        
        ///return program output type ('f', 'b')             
        vector<char> get_otypes();
        ///return parameter otype, used to set otypes
        char get_otype(){return params.otype;};
        
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
        float get_split();
        
        ///add custom node into feat
        /* void add_function(unique_ptr<Node> N)
         * { params.functions.push_back(N->clone()); } */
        
        ///return data types for input parameters
        vector<char> get_dtypes();

        ///return feedback setting
        float get_fb();
       
        ///return best model
        string get_representation();

        ///return best model, in tabular form
        string get_model(bool sort=true);

        ///return best model as a single line equation 
        string get_eqn(bool sort) ;

        ///get number of parameters in best
        int get_n_params();

        ///get dimensionality of best
        int get_dim();

        ///get dimensionality of best
        int get_complexity();

        ///return population as string
        string get_archive(bool front=true);
       
        /// return the coefficients or importance scores of the best model. 
        ArrayXf get_coefs();

        /// return the number of nodes in the best model
        int get_n_nodes();

        /// get longitudinal data from file s
        LongData get_Z(string s, 
                int * idx, int idx_size);

        string get_sel(){return this->selector.get_type();};
        void set_sel(string in){this->selector.set_type(in); };

        string get_surv(){return this->survivor.get_type();};
        void set_surv(string in){this->survivor.set_type(in); };

        bool get_tune_initial(){ return this->params.tune_initial;};
        void set_tune_initial(bool in){ this->params.tune_initial = in;};

        bool get_tune_final(){ return this->params.tune_final;};
        void set_tune_final(bool in){ this->params.tune_final = in;};

        string get_functions(){return params.get_functions();};
        string get_functions_(){return params.get_functions_();};

        string get_obj(){return params.get_objectives(); };  
        void set_obj(string in){return params.set_objectives(in); };  

        string get_protected_groups(){ return params.get_protected_groups(); };
        ///set protected groups for fairness
        void set_protected_groups(string pg);

        bool get_val_from_arch(){return val_from_arch; };
        void set_val_from_arch(bool in){val_from_arch = in; };


        /// destructor             
        ~Feat();
                    
        /// train a model.             
        void fit(MatrixXf& X, VectorXf& y, LongData Z = LongData());
                        
        void run_generation(unsigned int g,
                        vector<size_t> survivors,
                        DataRef &d,
                        std::ofstream &log,
                        float percentage,
                        unsigned& stall_count);
                 
        /// train a model.             
        void fit(float * X,int rowsX,int colsX, float * Y,int lenY);

        /// train a model, first loading longitudinal samples (Z) from file.
        void fit_with_z(float * X, int rowsX, int colsX, 
                float * Y, int lenY, string s, int * idx, int idx_size);
       
        /// predict on unseen data.             
        VectorXf predict(MatrixXf& X, LongData Z = LongData());  
        
        /// predict on unseen data from the whole archive             
        VectorXf predict_archive(int id, MatrixXf& X, 
                LongData Z = LongData());  
        VectorXf predict_archive(int id, float * X, int rowsX, int colsX);
        ArrayXXf predict_proba_archive(int id, MatrixXf& X, 
                LongData Z=LongData());

        ArrayXXf predict_proba_archive(int id, 
                float * X, int rows_x, int cols_x);
        /// predict on unseen data. return CLabels.
        shared_ptr<CLabels> predict_labels(MatrixXf& X,
                         LongData Z = LongData());  

        /// predict probabilities of each class.
        ArrayXXf predict_proba(MatrixXf& X,
                         LongData Z = LongData());  
        
        ArrayXXf predict_proba(float * X, int rows_x, int cols_x);

        /// predict on unseen data, loading longitudinal samples (Z) from file.
        VectorXf predict_with_z(float * X, int rowsX,int colsX, 
                                string s, int * idx, int idx_size);

        /// predict probabilities of each class.
        ArrayXXf predict_proba_with_z(float * X, int rowsX,int colsX, 
                                string s, int * idx, int idx_size);  

        /// predict on unseen data.             
        VectorXf predict(float * X, int rowsX, int colsX);      
        
        /// transform an input matrix using a program.                          
        MatrixXf transform(MatrixXf& X,
                           LongData Z = LongData(),
                           Individual *ind = 0);
        
        MatrixXf transform(float * X,  int rows_x, int cols_x);
        
        /// train a model, first loading longitudinal samples (Z) from file.
        MatrixXf transform_with_z(float * X, int rowsX, int colsX, string s, 
                                  int * idx, int idx_size);
        
        /// convenience function calls fit then predict.            
        VectorXf fit_predict(MatrixXf& X,
                             VectorXf& y,
                             LongData Z = LongData());
                             
        VectorXf fit_predict(float * X, int rows_x, int cols_x, float * Y, 
                int len_y);
        
        /// convenience function calls fit then transform. 
        MatrixXf fit_transform(MatrixXf& X,
                               VectorXf& y,
                               LongData Z = LongData());
                               
        MatrixXf fit_transform(float * X, int rows_x, int cols_x, float * Y, 
                int len_y);
              
        /// scoring function 
        float score(MatrixXf& X, const VectorXf& y,
                 LongData Z = LongData()); 
        
        /// return statistics from the run as a json string
        string get_stats();

        /// load best_ind from file
        void load_best_ind(string filename);

        /// load population from file, optionall just Pareto front
        void load_population(string filename, bool justfront=false);

        /// load Feat state from a json string.
        void load(const string& feat_state);
        /// load Feat state from file.
        void load_from_file(string filename);
        /// save and return a json Feat state as string.
        string save();
        /// save Feat state to file.
        void save_to_file(string filename);
        
        bool fitted; ///< keeps track of whether fit was called.
    private:
        // Parameters
        Parameters params;  ///< hyperparameters of Feat 

        Timer timer;       ///< start time of training
        // subclasses for main steps of the evolutionary routine
        Population pop;       	///< population of programs
        Selection selector;        	///< selection algorithm
        Evaluation evaluator;      	///< evaluation code
        Variation variator;  	///< variation operators
        Selection survivor;       	///< survival algorithm
        Archive archive;          ///< pareto front archive
        bool use_arch;         ///< internal control over use of archive
        string survival;                        ///< stores survival mode
        Normalizer N;                           ///< scales training data.
        // performance tracking
        float min_loss;                      ///< current best score
        float min_loss_v;                    ///< best validation score
        float best_med_score;  ///< best median population score
        int best_complexity;  ///< complexity of the best model
        string str_dim; ///< dimensionality as multiple of number of cols 
        string starting_pop; ///< file with starting population
        Individual best_ind;                    ///< best individual
        string logfile;                         ///< log filename
        int save_pop;  ///< controls whether pop is printed each gen
        bool val_from_arch; ///< model selection only uses Pareto front
        float simplify;  ///< post-run simplification
        Log_Stats stats; ///< runtime stats
        int random_state;

        /* functions */
        /// updates best score
        bool update_best(const DataRef& d, bool val=false);    
        
        /// calculate and print stats
        void calculate_stats(const DataRef& d);
        void print_stats(std::ofstream& log,
                         float fraction);      

        // gets weights via univariate initial models
        vector<float> univariate_initial_model(DataRef &d, int n_feats);
        /// method to fit inital ml model            
        void initial_model(DataRef &d);
        /// fits final model to best transformation
        void final_model(DataRef& d);
        /// simplifies final model to best transformation
        void simplify_model(DataRef& d, Individual&);
        /// updates stall count for early stopping
        void update_stall_count(unsigned& stall_count, bool updated);
        
        //serialization
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(Feat,
                params,
                pop,
                selector,
                survivor,
                archive,
                use_arch,
                survival,
                N,
                min_loss,
                min_loss_v,
                best_med_score,
                best_complexity,
                str_dim,
                starting_pop,
                best_ind,
                fitted,
                random_state
                );
                
};

// serialization
} // FT
#endif
