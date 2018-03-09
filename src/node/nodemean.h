/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_MEAN
#define NODE_MEAN

#include "node.h"

namespace FT{
	class NodeMean : public Node
    {
    	public:
    	
    		NodeMean()
            {
                name = "mean";
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
                ArrayXd tmp(stack.l.top().size());
                
                int x;
                
                for(x = 0; x < stack.l.top().size(); x++)
                    tmp(x) = stack.l.top()[x].mean();
                    
                stack.l.pop();

                stack.f.push(tmp);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                string x1 = stack.ls.pop();
                stack.fs.push("mean(" + x1 + ")");
            }
    };
}	

#endif
