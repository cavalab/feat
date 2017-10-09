/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_NOT
#define NODE_NOT

#include "node.h"

namespace FT{
	class NodeNot : public Node
    {
    	public:
    	
    		NodeNot()
    		{
    			std::cerr << "error in nodenot.h : invalid constructor called";
				throw;
    		}
    	
    		NodeNot(string n)
    		{
    			name = n;
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 1;
    			complexity = 1;
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
