/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GREATERTHAN
#define NODE_GREATERTHAN

#include "node.h"

namespace FT{
	class NodeGreaterThan : public Node
    {
    	public:
    	
    		NodeGreaterThan()
    		{
    			std::cerr << "error in nodegreaterthan.h : invalid constructor called";
				throw;
    		}
    	
    		NodeGreaterThan(string n)
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
