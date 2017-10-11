/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_DIVIDE
#define NODE_DIVIDE

#include "node.h"

namespace FT{
	class NodeDivide : public Node
    {
    	public:
    	  	
    		NodeDivide()
    		{
    			name = "/";
    			otype = 'f';
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
            	if (stack_f.size() >= arity['f'] && stack_b.size() >= arity['b'])
            	{
            		ArrayXd x2 = stack_f.back(); stack_f.pop_back();
                    ArrayXd x1 = stack_f.back(); stack_f.pop_back();

                    stack_f.push_back(x1 / x2);
            	}
            }

            /*!
             * @brief evaluates the node symbolically
             */
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
            	if (stack_f.size() >= arity['f'] && stack_b.size() >= arity['b'])
            	{
            		string x2 = stack_f.back(); stack_f.pop_back();
                    string x1 = stack_f.back(); stack_f.pop_back();
                    stack_f.push_back("(" + x1 + "/" + x2 + ")");            	
            	}
            }
    };
}	

#endif
