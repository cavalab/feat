/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_EQUAL
#define NODE_EQUAL

#include "node.h"

namespace FT{
	class NodeEqual : public Node
    {
    	public:
    	   	
    		NodeEqual()
    		{
    			name = "=";
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
            	ArrayXd x2 = stack.f.pop();
                ArrayXd x1 = stack.f.pop();
                stack.b.push(x1 == x2);
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
            	string x2 = stack.fs.pop();
                string x1 = stack.fs.pop();
                stack.bs.push("(" + x1 + "==" + x2 + ")");
            }
    };
}	

#endif
