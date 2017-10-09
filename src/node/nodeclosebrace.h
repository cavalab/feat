/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CLOSEBRACE
#define NODE_CLOSEBRACE

#include "node.h"

namespace FT{
	class NodeCloseBrace : public Node
    {
    	public:
    	
    		NodeCloseBrace()
    		{
    			std::cerr << "error in nodeclosebrace.h : invalid constructor called";
				throw;
    		}
    	
    		NodeCloseBrace(string n)
    		{
    			name = n;
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		
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

#endif
