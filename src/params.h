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
        bool classification;            			///< flag to conduct classification rather than 
        int max_stall;                  			///< maximum stall in learning, in generations
        char otype;                     			///< program output type ('f', 'b')
        int verbosity;                  			///< amount of printing. 0: none, 1: minimal, 
                                                    // 2: all
        vector<double> term_weights;    			///< probability weighting of terminals
        vector<std::shared_ptr<Node>> functions;    ///< function nodes available in programs
        vector<std::shared_ptr<Node>> terminals;    ///< terminal nodes available in programs
        unsigned int max_depth;         			///< max depth of programs
        unsigned int max_size;          			///< max size of programs (length)
        unsigned int max_dim;           			///< maximum dimensionality of programs
        bool erc;								    ///<whether to include constants for terminals 
        vector<string> objectives;                  ///< Pareto objectives 

        Parameters(int pop_size, int gens, string& ml, bool classification, int max_stall, 
                   char otype, int vebosity, string functions, unsigned int max_depth, 
                   unsigned int max_dim, bool constant, string obj):    
            pop_size(pop_size),
            gens(gens),
            ml(ml),
            classification(classification),
            max_stall(max_stall),
            otype(otype), 
            verbosity(verbosity),
            max_depth(max_depth),
            max_dim(max_dim),
            erc(constant)
        {
            set_functions(functions);
            set_objectives(obj);
            max_size = pow(2,max_depth)*max_dim; // max_size is max_dim binary trees of max_depth
        }
        
        
        ~Parameters(){}

        /*!
         * @brief print message with verbosity control. 
         */
        void msg(string m, int v) const
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
        std::shared_ptr<Node> createNode(std::string str, double d_val = 0, bool b_val = false, 
                                         size_t loc = 0);
        
        /*!
         * @brief sets available functions based on comma-separated list.
         */
        void set_functions(string fs);
        
        /*!
         * @brief set the terminals
         */
        void set_terminals(int num_features);

        /// set the objectives
        void set_objectives(string obj);

    };

    /////////////////////////////////////////////////////////////////////////////////// Definitions
    
    std::shared_ptr<Node> Parameters::createNode(string str, double d_val, bool b_val, size_t loc)
    {
        // algebraic operators
    	if (str.compare("+") == 0) 
    		return std::shared_ptr<Node>(new NodeAdd());
        
        else if (str.compare("-") == 0)
    		return std::shared_ptr<Node>(new NodeSubtract());

        else if (str.compare("*") == 0)
    		return std::shared_ptr<Node>(new NodeMultiply());

     	else if (str.compare("/") == 0)
    		return std::shared_ptr<Node>(new NodeDivide());

        else if (str.compare("sqrt") == 0)
    		return std::shared_ptr<Node>(new NodeSqrt());
    	
    	else if (str.compare("sin") == 0)
    		return std::shared_ptr<Node>(new NodeSin());
    		
    	else if (str.compare("cos") == 0)
    		return std::shared_ptr<Node>(new NodeCos());
    	   
        else if (str.compare("^2") == 0)
    		return std::shared_ptr<Node>(new NodeSquare());
 	
        else if (str.compare("^3") == 0)
    		return std::shared_ptr<Node>(new NodeCube());
    	
        else if (str.compare("^") == 0)
    		return std::shared_ptr<Node>(new NodeExponent());

        else if (str.compare("exp") == 0)
    		return std::shared_ptr<Node>(new NodeExponential());

        else if (str.compare("log") == 0)
    		return std::shared_ptr<Node>(new NodeLog());   

        // logical operators
        else if (str.compare("and") == 0)
    		return std::shared_ptr<Node>(new NodeAnd());
       
    	else if (str.compare("or") == 0)
    		return std::shared_ptr<Node>(new NodeOr());
   		
     	else if (str.compare("not") == 0)
    		return std::shared_ptr<Node>(new NodeNot());
   		
    	else if (str.compare("=") == 0)
    		return std::shared_ptr<Node>(new NodeEqual());
    		
        else if (str.compare(">") == 0)
    		return std::shared_ptr<Node>(new NodeGreaterThan());

    	else if (str.compare(">=") == 0)
    		return std::shared_ptr<Node>(new NodeGEQ());        

    	else if (str.compare("<") == 0)
    		return std::shared_ptr<Node>(new NodeLessThan());
    	
    	else if (str.compare("<=") == 0)
    		return std::shared_ptr<Node>(new NodeLEQ());
    	
     	else if (str.compare("if") == 0)
    		return std::shared_ptr<Node>(new NodeIf());   	    		
        	
    	else if (str.compare("ite") == 0)
    		return std::shared_ptr<Node>(new NodeIfThenElse());

        // variables and constants
        else if (str.compare("x") == 0)
            return std::shared_ptr<Node>(new NodeVariable(loc));
            
        else if (str.compare("kb")==0)
            return std::shared_ptr<Node>(new NodeConstant(b_val));
            
        else if (str.compare("kd")==0)
            return std::shared_ptr<Node>(new NodeConstant(d_val));
            
        else
        {
            std::cerr << "Error: no node named " << str << " exists.\n"; 
            throw;
        }
        //TODO: add squashing functions, time delay functions, and stats functions
    	
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
        fs += ',';          // add delimiter to end 
        string delim = ",";
        size_t pos = 0;
        string token;
        while ((pos = fs.find(delim)) != string::npos) 
        {
            token = fs.substr(0, pos);
            msg("setting function with token "+token,0);
            functions.push_back(createNode(token));
            fs.erase(0, pos + delim.length());
        } 
        // print    
        string toprint = "functions: ";
        for (const auto& f : functions)
            toprint += f->name + ' ';
        msg(toprint,0);
    }

    void Parameters::set_terminals(int num_features)
    {
        /*!
         * based on number of features.
         */
        
        for (size_t i = 0; i < num_features; ++i)
            terminals.push_back(createNode(string("x"), 0, 0, i));
    	
        if(erc)
    		for (int i = 0; i < num_features; ++i)
    		{
    			if(r() < 0.5)
    	       		terminals.push_back(createNode(string("kb"), 0, r(), 0));
    	       	else
    	       		terminals.push_back(createNode(string("kd"), r(), 0, 0));
    	    }                
    }

    void Parameters::set_objectives(string obj)
    {
        /*! Input: obj, a comma-separated list of objectives
         */

        fs += ',';          // add delimiter to end 
        string delim = ",";
        size_t pos = 0;
        string token;
        while ((pos = fs.find(delim)) != string::npos) 
        {
            token = fs.substr(0, pos);
            objectives.push_back(token);
            fs.erase(0, pos + delim.length());
        }
    }
}
#endif
