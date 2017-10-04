/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARAMS_H
#define PARAMS_H
// internal includes
#include "node.h"

namespace FT{

    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Parameters
     * @brief holds the hyperparameters for Fewtwo. 
     */
    struct Parameters
    {
        int pop_size;                   ///< population size
        int gens;                       ///< max generations
        string ml;                      ///< machine learner with which Fewtwo is paired
        bool classification;            ///< flag to conduct classification rather than regression
        int max_stall;                  ///< maximum stall in learning, in generations
        char otype;                     ///< output type of the programs ('f': float, 'b': boolean)
        int verbosity;                  ///< amount of printing. 0: none, 1: minimal, 2: all
        vector<double> term_weights;    ///< probability weighting of terminals
        vector<Node> functions;         ///< function nodes available in programs
        vector<Node> terminals;         ///< terminal nodes available in programs
        unsigned int max_depth;         ///< max depth of programs
        unsigned int max_size;          ///< max size of programs (length)
        unsigned int max_dim;           ///< maximum dimensionality of programs 

        Parameters(int pop_size, int gens, string& ml, bool classification, int max_stall, 
                   char otype, int vebosity, string functions, unsigned int max_depth, 
                   unsigned int max_dim):    
            pop_size(pop_size),
            gens(gens),
            ml(ml),
            classification(classification),
            max_stall(max_stall),
            otype(otype), 
            verbosity(verbosity),
            max_depth(max_depth),
            max_dim(max_dim)
        {
            set_functions(functions);
            max_size = pow(2,max_depth)*max_dim; // max_size is max_dim binary trees of max_depth
        }
        
        
        ~Parameters(){}

        /*!
         * @brief print message with verbosity control. 
         */
        void msg(string m, int v)
        {
            /* prints messages based on verbosity level. */

            if (verbosity >= v)
                std::cout << m << "...\n";
        }
        
        /*!
         * @brief sets weights for terminals. 
         */
        void set_term_weights(const vector<double>& w)
        {
            std::cout << "w size: " << w.size() << "\n";
            std::cout << "terminals size: " << terminals.size() << "\n";
            assert(w.size()==terminals.size());
            term_weights = w; 
        }
        
        /*!
         * @brief sets available functions based on comma-separated list.
         */
        void set_functions(string fs);
        
        /*!
         * @brief set the terminals
         */
        void set_terminals(int num_features);
    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions

    void Parameters::set_functions(string fs)
    {
        /*! 
         * Input: 
         *
         *		fs: string of comma-separated Node names
         *
         * Output:
         *
         *		modifies functions 
         *
         */
        string delim = ",";
        size_t pos = 0;
        string token;
        while ((pos = fs.find(delim)) != string::npos) 
        {
            token = fs.substr(0, pos);
            functions.push_back(Node((*token.c_str())));
            fs.erase(0, pos + delim.length());
        } 
    }

    void Parameters::set_terminals(int num_features)
    {
        /*!
         * based on number of features.
         */
        for (size_t i = 0; i < num_features; ++i) 
            terminals.push_back(Node('x',i)); 
    }
}
#endif
