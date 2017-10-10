/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EQUAL
#define NODE_EQUAL

#include "node.h"

namespace FT{
	class NodeEqual : public Node
    {
    	public:
    	   	
    		NodeEqual()
    		{
    			name = "=";
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 1;
    		}
    		/*!
             * @brief Evaluates the node and updates the stack states. 
             */
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
            	ArrayXb x = stack_b.back(); stack_b.pop_back();
                ArrayXb y = stack_b.back(); stack_b.pop_back();
                stack_b.push_back(x == y);
            }

            /*!
             * @brief evaluates the node symbolically
             */
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	string x = stack_b.back(); stack_b.pop_back();
                string y = stack_b.back(); stack_b.pop_back();
                stack_b.push_back("(" + x + "==" + y + ")");
            }
    };
}	

#endif
