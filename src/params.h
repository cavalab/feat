/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef PARAMS_H
#define PARAMS_H
// internal includes
#include "nodewrapper.h"

namespace FT{

    ////////////////////////////////////////////////////////////////////////////////// Declarations
    /*!
     * @class Parameters
     * @brief holds the hyperparameters for Fewtwo. 
     */
    struct Parameters
    {
        int pop_size;                   			///< population size
        int gens;                       			///< max generations
        string ml;                      			///< machine learner with which Fewtwo is paired
        bool classification;            			///< flag to conduct classification rather than regression
        int max_stall;                  			///< maximum stall in learning, in generations
        char otype;                     			///< output type of the programs ('f': float, 'b': boolean)
        int verbosity;                  			///< amount of printing. 0: none, 1: minimal, 2: all
        vector<double> term_weights;    			///< probability weighting of terminals
        vector<std::shared_ptr<Node>> functions;    ///< function nodes available in programs
        vector<std::shared_ptr<Node>> terminals;    ///< terminal nodes available in programs
        unsigned int max_depth;         			///< max depth of programs
        unsigned int max_size;          			///< max size of programs (length)
        unsigned int max_dim;           			///< maximum dimensionality of programs 

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
         * @brief return shared pointer to a node based on the string passed
         */
        std::shared_ptr<Node> createNode(std::string str, double d_val = 0, bool b_val = false, size_t loc = 0);
        
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
    
    std::shared_ptr<Node> Parameters::createNode(string str, double d_val, bool b_val, size_t loc)
    {
    	if (str.compare("+") == 0)
    	{
    		return std::shared_ptr<Node>(new NodeAdd(str));
    	}
    	if (str.compare("and") == 0)
    		return std::shared_ptr<Node>(new NodeAnd(str));
    		
    	if (str.compare("}") == 0)
    		return std::shared_ptr<Node>(new NodeCloseBrace(str));
    		
    	if (str.compare("doubleConst") == 0)
    		return std::shared_ptr<Node>(new NodeConstant(str, d_val));
    		
    	if (str.compare("boolConst") == 0)
    		return std::shared_ptr<Node>(new NodeConstant(str, b_val));
    		
    	if (str.compare("cos") == 0)
    		return std::shared_ptr<Node>(new NodeCosine(str));
    		
    	if (str.compare("cube") == 0)
    		return std::shared_ptr<Node>(new NodeCube(str));
    		
    	if (str.compare("/") == 0)
    		return std::shared_ptr<Node>(new NodeDivide(str));
    		
    	if (str.compare("=") == 0)
    		return std::shared_ptr<Node>(new NodeEqual(str));
    		
    	if (str.compare("^") == 0)
    		return std::shared_ptr<Node>(new NodeExponent(str));
    		
    	if (str.compare("exp") == 0)
    		return std::shared_ptr<Node>(new NodeExponential(str));
    		
    	if (str.compare(">") == 0)
    		return std::shared_ptr<Node>(new NodeGreaterThan(str));
    		
    	if (str.compare("if") == 0)
    		return std::shared_ptr<Node>(new NodeIf(str));
    		
    	if (str.compare("<") == 0)
    		return std::shared_ptr<Node>(new NodeLessThan(str));
    		
    	if (str.compare("log") == 0)
    		return std::shared_ptr<Node>(new NodeLog(str));
    		
    	if (str.compare("*") == 0)
    		return std::shared_ptr<Node>(new NodeMultiply(str));
    		
    	if (str.compare("!") == 0)
    		return std::shared_ptr<Node>(new NodeNot(str));
    	
    	if (str.compare("{") == 0)
    		return std::shared_ptr<Node>(new NodeOpenBrace(str));
    	
    	if (str.compare("|") == 0)
    		return std::shared_ptr<Node>(new NodeOr(str));
    		
    	if (str.compare("sqroot") == 0)
    		return std::shared_ptr<Node>(new NodeRoot(str));
    	
    	if (str.compare("sin") == 0)
    		return std::shared_ptr<Node>(new NodeSine(str));
    		
    	if (str.compare("square") == 0)
    		return std::shared_ptr<Node>(new NodeSquare(str));
    	
    	if (str.compare("-") == 0)
    		return std::shared_ptr<Node>(new NodeSubtract(str));
    	
    	if (str.compare("then") == 0)
    		return std::shared_ptr<Node>(new NodeThen(str));
    	
    	if (str.compare("variable") == 0)
    		return std::shared_ptr<Node>(new NodeVariable(str, loc));
    }

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
            functions.push_back(createNode(token));
            fs.erase(0, pos + delim.length());
        } 
    }

    void Parameters::set_terminals(int num_features)
    {
        /*!
         * based on number of features.
         */
        for (size_t i = 0; i < num_features; ++i) 
            terminals.push_back(createNode(string("variable"), 0, 0, i)); 
    }
}
#endif
