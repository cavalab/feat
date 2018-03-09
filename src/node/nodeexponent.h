/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EXPONENT
#define NODE_EXPONENT

#include "node.h"

namespace FT{
	class NodeExponent : public Node
    {
    	public:
    	  	
    		NodeExponent()
    		{
    			name = "^";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 4;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
           		ArrayXd x2 = stack.f.pop();
                ArrayXd x1 = stack.f.pop();

                stack.f.push(limited(pow(x1,x2)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x2 = stack.fs.pop();
                string x1 = stack.fs.pop();
                stack.fs.push("(" + x1 + ")^(" + x2 + ")");
            }
    };
}	

#endif
