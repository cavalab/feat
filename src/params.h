/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARAMS_H
#define PARAMS_H
// internal includes
#include "nodewrapper.h"
#include "nodevector.h"

#include "utils.h"

namespace FT{

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
        vector<double> term_weights;    			///< probability weighting of terminals
        NodeVector functions;                       ///< function nodes available in programs
        NodeVector terminals;                       ///< terminal nodes available in programs
        vector<std::string> longitudinalMap;        ///<vector storing longitudinal data keys

        unsigned int max_depth;         			///< max depth of programs
        unsigned int max_size;          			///< max size of programs (length)
        unsigned int max_dim;           			///< maximum dimensionality of programs
        bool erc;								    ///<whether to include constants for terminals 
        unsigned num_features;                      ///< number of features
        vector<string> objectives;                  ///< Pareto objectives 
        bool shuffle;                               ///< option to shuffle the data
        double split;                               ///< fraction of data to use for training
        vector<char> dtypes;                        ///< data types of input parameters
        double feedback;                            ///< strength of ml feedback on probabilities
        unsigned int n_classes;                     ///< number of classes for classification 
        float cross_rate;                           ///< cross rate for variation
        vector<int> classes;                        ///< class labels
        vector<float> class_weights;                ///< weights for each class
        vector<float> sample_weights;               ///< weights for each sample 
        string scorer;                              ///< loss function
        vector<string> feature_names;               ///< names of features
        bool backprop;                              ///< turns on backpropagation
        bool hillclimb;                             ///< turns on parameter hill climbing

        struct BP 
        {
           int iters;
           double learning_rate;
           int batch_size;
           BP(int i, double l, int bs): iters(i), learning_rate(l), batch_size(bs) {}
        };

        BP bp;                                      ///< backprop parameters
        
        struct HC 
        {
           int iters;
           double step;
           HC(int i, double s): iters(i), step(s) {}
        };
        
        HC hc;                                      ///< stochastic hill climbing parameters       
        
        Parameters(int pop_size, int gens, string ml, bool classification, int max_stall, 
                   char ot, int verbosity, string fs, float cr, unsigned int max_depth, 
                   unsigned int max_dim, bool constant, string obj, bool sh, double sp, 
                   double fb, string sc, string fn, bool bckprp, int iters, double lr,
                   int bs, bool hclimb);
        
        ~Parameters();
        
        /*! checks initial parameter settings before training.
         *  make sure ml choice is valid for problem type.
         *  make sure scorer is set. 
         *  for classification, check clases and find number.
         */
        void init();
        
        /// print message with verbosity control. 
        string msg(string m, int v, string sep="\n") const;
      
        /// sets current generation
        void set_current_gen(int g);
        
        /// sets scorer type
        void set_scorer(string sc);
        
        /// sets weights for terminals. 
        void set_term_weights(const vector<double>& w);
        
        /// return shared pointer to a node based on the string passed
        std::unique_ptr<Node> createNode(std::string str, double d_val = 0, bool b_val = false, 
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
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z = 
                           std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > ());

        void set_feature_names(string fn); 
        /// set the objectives
        void set_objectives(string obj);
        
        /// set level of debug info
        void set_verbosity(int verbosity);

        void set_otype(char ot);
        
        void set_ttypes();

        /// set the output types of programs
        void set_otypes();
        
        /// sets the number of classes based on target vector y.
        void set_classes(VectorXd& y);
        
        void set_sample_weights(VectorXd& y);
    };
}
#endif
