/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SKEW
#define NODE_SKEW

#include "node.h"

namespace FT{
	class NodeSkew : public Node
    {
    	public:
    	
    		NodeSkew()
            {
                name = "skew";
    			otype = 'f';
    			arity['f'] = 0;
    			arity['b'] = 0;
    			arity['l'] = 1;
    			complexity = 3;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
                ArrayXd tmp(stack.l.top().size());
                
                int x;
                
                for(x = 0; x < stack.l.top().size(); x++)
                    tmp(x) = skew(stack.l.top()[x]);
                    
                stack.l.pop();

                stack.f.push(tmp);
                
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                string x1 = stack.ls.pop();
                stack.fs.push("skew(" + x1 + ")");
            }
    };
}	

#endif
