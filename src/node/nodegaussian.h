/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_GAUSSIAN
#define NODE_GAUSSIAN

#include "node.h"

namespace FT{
	class NodeGaussian : public Node
    {
    	public:
    	
    		NodeGaussian()
            {
                name = "gaussian";
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
                stack.f.push(limited(exp(-1*limited(pow(x, 2)))));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x = stack.fs.pop();
                stack.fs.push("exp(-(" + x + " ^ 2))");
            }
    };
}	

#endif
