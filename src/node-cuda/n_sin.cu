/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SIN
#define NODE_SIN

#include "node.h"

namespace FT{
	class NodeSin : public Node
    {
    	public:
    	
    		NodeSin()
       		{
    			name = "sin";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 3;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, vector<ArrayXd>& stack_f, vector<ArrayXb>& stack_b)
            {

        		ArrayXd x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back(limited(sin(x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(vector<string>& stack_f, vector<string>& stack_b)
            {
        		string x = stack_f.back(); stack_f.pop_back();
                stack_f.push_back("sin(" + x + ")");
            }
    };
}	

#endif
