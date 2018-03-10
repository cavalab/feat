/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MAX
#define NODE_MAX

#include "node.h"

namespace FT{
	class NodeMax : public Node
    {
    	public:
    	
    		NodeMax()
            {
                name = "max";
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['l'] = 1;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
                ArrayXd tmp(stack.z.top().size());
                
                int x;
                
                for(x = 0; x < stack.z.top().size(); x++)
                    tmp(x) = stack.z.top()[x].maxCoeff();
                    
                stack.z.pop();

                stack.f.push(tmp);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("max(" + stack.zs.pop() + ")");
            }
    };
}	

#endif
