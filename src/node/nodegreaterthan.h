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
    			name = ">";
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		/*!
             * @brief Evaluates the node and updates the stack states. 
             */
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, 
                    vector<ArrayXb>& stack_b)
            {
                ArrayXf x = stack_f.back(); stack_f.pop_back();
                ArrayXf y = stack_f.back(); stack_f.pop_back();
                stack_b.push_back(x > y);
            }

            /*!
             * @brief evaluates the node symbolically
             */
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	string x = stack_f.back(); stack_f.pop_back();
                string y = stack_f.back(); stack_f.pop_back();
                stack_b.push_back("(" + x + ">" + y + ")");
            }
    };
}	

#endif
