/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_AND
#define NODE_AND

#include "node.h"

namespace FT{
	class NodeAnd : public Node
    {
    	public:
    	
    		NodeAnd()
       		{
    			name = "and";
    			otype = 'b';
    			arity['f'] = 0;
    			arity['b'] = 2;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y, const vector<vector<ArrayXd> > &Z, 
			        Stacks& stack)
            {
                ArrayXb x2 = stack.b.pop();
                ArrayXb x1 = stack.b.pop();
                stack.b.push(x1 && x2);

            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                string x2 = stack.bs.pop();
                string x1 = stack.bs.pop();
                stack.bs.push("(" + x1 + " AND " + x2 + ")");
            }
    };
}	

#endif
