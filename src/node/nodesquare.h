/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SQUARE
#define NODE_SQUARE

#include "node.h"

namespace FT{
	class NodeSquare : public Node
    {
    	public:
    	
    		NodeSquare()
    		{
    			name = "^2";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 1;
    		}
    		/*!
             * @brief Evaluates the node and updates the stack states. 
             */
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
            	if (stack_f.size() >= arity['f'] && stack_b.size() >= arity['b'])
            	{
            		ArrayXd x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back(pow(x,2));
            	}
            }

            /*!
             * @brief evaluates the node symbolically
             */
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	if (stack_f.size() >= arity['f'] && stack_b.size() >= arity['b'])
            	{
            		string x = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("(" + x + "^2)");
            	}
            }
    };
}	

#endif
