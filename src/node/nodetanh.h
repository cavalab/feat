/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_TANH
#define NODE_TANH

#include "node.h"

namespace FT{
	class NodeTanh : public Node
    {
    	public:
    	
    		NodeTanh()
            {
                name = "tanh";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 3;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
			{
        		ArrayXd x = stack.f.pop();
                stack.f.push(limited(tanh(x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x = stack.fs.pop();
                stack.fs.push("tanh(" + x + ")");
            }
    };
}	

#endif
