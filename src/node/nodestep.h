/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_STEP
#define NODE_STEP

#include "node.h"

namespace FT{
	class NodeStep : public Node
    {
    	public:
    	
    		NodeStep()
            {
                name = "step";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
        		ArrayXd x = stack.f.pop();
        		
        		ArrayXd res = (x > 0).select(ArrayXd::Ones(x.size()), ArrayXd::Zero(x.size())); 
                stack.f.push(res);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x = stack.fs.pop();
                stack.fs.push("step("+ x +")");
            }
    };
}	

#endif
