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
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
        		ArrayXd x = stack.f.pop();
                stack.f.push(pow(x,2));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x = stack.fs.pop();
                stack.fs.push("(" + x + "^2)");
            }
    };
}	

#endif
