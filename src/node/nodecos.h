/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_COS
#define NODE_COS

#include "node.h"

namespace FT{
	class NodeCos : public Node
    {
    	public:
    	  	
    		NodeCos()
    		{
    			name = "cos";
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
                stack.f.push(limited(cos(x)));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x = stack.fs.pop();
                stack.fs.push("cos(" + x + ")");
            }
    };
}	

#endif
