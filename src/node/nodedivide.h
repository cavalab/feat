/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_DIVIDE
#define NODE_DIVIDE

#include "node.h"

namespace FT{
	class NodeDivide : public Node
    {
    	public:
    	  	
    		NodeDivide()
    		{
    			name = "/";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
                ArrayXd x2 = stack.f.pop();
                ArrayXd x1 = stack.f.pop();
                // safe division returns x1/x2 if x2 != 0, and MAX_DBL otherwise               
                stack.f.push( (abs(x2) > NEAR_ZERO ).select(x1 / x2, 1.0) ); //MAX_DBL    
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("(" + stack.fs.pop() + "/" + stack.fs.pop() + ")");            	
            }
    };
}	

#endif
