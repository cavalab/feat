/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EXPONENTIAL
#define NODE_EXPONENTIAL

#include "node.h"

namespace FT{
	class NodeExponential : public Node
    {
    	public:
   	
    		NodeExponential()
    		{
    			name = "exp";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 4;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
           		ArrayXd x = stack.f.pop();
                stack.f.push(limited(exp(x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x = stack.fs.pop();
                stack.fs.push("exp(" + x + ")");
            }
    };
}	

#endif
