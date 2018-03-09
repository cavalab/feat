/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_LOGIT
#define NODE_LOGIT

#include "node.h"

namespace FT{
	class NodeLogit : public Node
    {
    	public:
    	
    		NodeLogit()
            {
                name = "logit";
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
                stack.f.push(1/(1+(limited(exp(-1*x)))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x = stack.fs.pop();
                stack.fs.push("1/(1+exp(-1*" + x + "))");
            }
    };
}	

#endif
