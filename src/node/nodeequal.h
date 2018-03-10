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
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
                stack.b.push(stack.f.pop() == stack.f.pop());
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("(" + stack.fs.pop() + "==" + stack.fs.pop() + ")");
            }
    };
}	

#endif
