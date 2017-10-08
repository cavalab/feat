/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_OPENBRACE
#define NODE_OPENBRACE

#include "node.h"

namespace FT{
	class NodeOpenBrace : public Node
    {
    	public:
    	
    		NodeOpenBrace()
    		{
    			std::cerr << "error in node.h : invalid constructor called";
				throw;
    		}
    	
    		NodeOpenBrace(std::string n) : name(n),
    						     otype('b'),
    						     arity['f'](2),
    						     arity['b'](0),
    						     complexity(2) {}
    		
    		/*!
             * @brief Evaluates the node and updates the stack states. 
             */
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXi>& stack_b)
            {
            	std::cerr << "invalid operator name\n";
            }

            /*!
             * @brief evaluates the node symbolically
             */
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	std::cerr << "invalid operator name\n";
            }
    };
}	
