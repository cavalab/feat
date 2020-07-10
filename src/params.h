/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARAMS_H
#define PARAMS_H
// internal includes
#include "pop/nodewrapper.h"
#include "pop/nodevector.h"
#include "util/logger.h"
#include "util/utils.h"

namespace FT{

    using namespace Pop;
    using namespace Op;

    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Parameters
     * @brief holds the hyperparameters for Feat. 
     */
    struct Parameters
    {
        int pop_size;                   			///< population size
        int gens;                       			///< max generations
        int current_gen;                            ///< holds current generation
        string ml;                      			///< machine learner used with Feat
        bool classification;            			///< flag to conduct classification rather than 
        int max_stall;                  			///< maximum stall in learning, in generations
        vector<char> otypes;                     	///< program output types ('f', 'b')
        vector<char> ttypes;                     	///< program terminal types ('f', 'b')
        char otype;                                 ///< user parameter for output type setup
        int verbosity;                  			///< amount of printing. 0: none, 1: minimal, 
                                                    // 2: all
        vector<float> term_weights;    			    ///< probability weighting of terminals
        vector<float> op_weights;    			    ///< probability weighting of functions
        NodeVector functions;                       ///< function nodes available in programs
        NodeVector terminals;                       ///< terminal nodes available in programs
        vector<std::string> longitudinalMap;        ///<vector storing longitudinal data keys

        unsigned int max_depth;         			///< max depth of programs
        unsigned int max_size;          			///< max size of programs (length)
        unsigned int max_dim;           			///< maximum dimensionality of programs
        bool erc;								    ///< whether to include constants for terminals 
        unsigned num_features;                      ///< number of features
        vector<string> objectives;                  ///< Pareto objectives 
        bool shuffle;                               ///< option to shuffle the data
        float split;                               ///< fraction of data to use for training
        vector<char> dtypes;                        ///< data types of input parameters
        float feedback;                            ///< strength of ml feedback on probabilities
        unsigned int n_classes;                     ///< number of classes for classification 
        float cross_rate;                           ///< cross rate for variation
        vector<int> classes;                        ///< class labels
        vector<float> class_weights;                ///< weights for each class
        vector<float> sample_weights;               ///< weights for each sample 
        string scorer;                              ///< loss function
        vector<string> feature_names;               ///< names of features
        bool backprop;                              ///< turns on backpropagation
        bool hillclimb;                             ///< turns on parameter hill climbing
        int max_time;                               ///< max time for fit method
        bool use_batch;                             ///< whether to use mini batch for training
        bool residual_xo;                           ///< use residual crossover  
        bool stagewise_xo;                          ///< use stagewise crossover  
        bool stagewise_xo_tol;                      ///< use stagewise crossover  
        bool corr_delete_mutate;                    ///< use correlation delete mutation   
        float root_xo_rate;                         ///<  crossover  
        bool softmax_norm;                          ///< use softmax norm on probabilities
        bool normalize;                             ///< whether to normalize the input data
        vector<bool> protected_groups;          ///<protected attributes in X
        bool tune_initial; ///< tune initial ML model
        bool tune_final; ///< tune final ML model
        
        struct BP 
        {
           int iters;
           float learning_rate;
           int batch_size;
           BP(int i, float l, int bs): iters(i), learning_rate(l), batch_size(bs) {}
        };

        BP bp;                                      ///< backprop parameters
        
        struct HC 
        {
           int iters;
           float step;
           HC(int i, float s): iters(i), step(s) {}
        };
        
        HC hc;                                      ///< stochastic hill climbing parameters       
        
        Parameters(int pop_size, int gens, string ml, bool classification, 
                int max_stall, char ot, int verbosity, string fs, float cr, 
                float root_xor, unsigned int max_depth, unsigned int max_dim, 
                bool constant, string obj, bool sh, float sp, float fb, 
                string sc, string fn, bool bckprp, int iters, float lr, int bs, 
                bool hclimb, int maxt, bool res_xo, bool stg_xo, 
                bool stg_xo_tol, bool sftmx, bool nrm, bool corr_mut, 
                bool tune_init, bool tune_fin);
        
        ~Parameters();
        
        /*! checks initial parameter settings before training.
         *  make sure ml choice is valid for problem type.
         *  make sure scorer is set. 
         *  for classification, check clases and find number.
         */
        void init(const MatrixXf& X, const VectorXf& y, string scorer);
      
        /// sets current generation
        void set_current_gen(int g);
        
        /// sets scorer type
        void set_scorer(string sc);
        
        /// sets weights for terminals. 
        void set_term_weights(const vector<float>& w);
        
        /// sets weights for operators. 
        void set_op_weights();
        
        /// return unique pointer to a node based on the string passed
        std::unique_ptr<Node> createNode(std::string str, float d_val = 0, bool b_val = false, 
                                         size_t loc = 0, string name = "");
        
        /// sets available functions based on comma-separated list.
        void set_functions(string fs);
        
        /// max_size is max_dim binary trees of max_depth
        void updateSize();
        
        /// set max depth of programs
        void set_max_depth(unsigned int max_depth);
        
        /// set maximum dimensionality of programs
        void set_max_dim(unsigned int max_dim);
        
        /// set the terminals
        void set_terminals(int nf,
                           std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > Z = 
                           std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > ());

        void set_feature_names(string fn); 
        
        void set_protected_groups(string fn); 
        /// set the objectives
        void set_objectives(string obj);
        
        /// set level of debug info
        void set_verbosity(int verbosity);

        void set_otype(char ot);
        
        void set_ttypes();

        /// set the output types of programs
        void set_otypes(bool terminals_set=false);
        
        /// sets the number of classes based on target vector y.
        void set_classes(const VectorXf& y);    
        
        /// sets the weights of each sample (and class weights)
        void set_sample_weights(VectorXf& y);

    };
}
#endif
