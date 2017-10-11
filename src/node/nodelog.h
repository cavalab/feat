/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOG
#define NODE_LOG

#include "node.h"

namespace FT{
	class NodeLog : public Node
    {
    	public:
    	
    		NodeLog()
       		{
    			name = "log";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;
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
                    stack_f.push_back(log(x));
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
                    stack_f.push_back("log(" + x + ")");
            	}
            }
    };
}	

#endif
