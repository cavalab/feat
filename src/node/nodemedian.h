/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MEDIAN
#define NODE_MEDIAN

#include "node.h"

namespace FT{
	class NodeMedian : public Node
    {
    	public:
    	
    		NodeMedian()
            {
                name = "median";
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
                    tmp(x) = median(stack.z.top()[x]);
                    
                stack.z.pop();

                stack.f.push(tmp);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                string x1 = stack.zs.pop();
                stack.fs.push("median(" + stack.zs.pop() + ")");
            }
    };
}	

#endif
