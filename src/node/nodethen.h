/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_THEN
#define NODE_THEN

#include "node.h"

namespace FT{
	class NodeThen : public Node
    {
    	public:
    	
    		NodeThen()
    		{
    			std::cerr << "error in nodethen.h : invalid constructor called";
				throw;
    		}
    	
    		NodeThen(string n)
    		{
    			name = n;
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 2;
    			complexity = 5;
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
