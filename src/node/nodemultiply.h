/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MULTIPLY
#define NODE_MULTIPLY

#include "node.h"

namespace FT{
	class NodeMultiply : public Node
    {
    	public:
    	
    		NodeMultiply()
       		{
    			name = "*";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}

            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
                stack.f.push(limited(stack.f.pop() * stack.f.pop()));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
            	stack.fs.push("(" + stack.fs.pop() + "*" + stack.fs.pop() + ")");
            }
    };
}	

#endif
