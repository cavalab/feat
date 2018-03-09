/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_CUBE
#define NODE_CUBE

#include "node.h"

namespace FT{
	class NodeCube : public Node
    {
    	public:
    		  
    		NodeCube()
    		{
    			name = "cube";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 33;
    		}
    		
            /// Evaluates the node and updates the stack states.  
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
        		ArrayXd x = stack.f.pop();
                stack.f.push(pow(x,3));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
        		string x = stack.fs.pop();
                stack.fs.push("(" + x + "^3)");
            }
    };
}	

#endif

