/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_OPENBRACE
#define NODE_OPENBRACE

#include "node.h"

namespace FT{
	class NodeLEQ : public Node
    {
    	public:
    	
    		NodeLEQ()
    		{
    			name = "<=";
    			otype = 'b';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
              	ArrayXd x1 = stack.f.pop();
                ArrayXd x2 = stack.f.pop();
                stack.b.push(x1 <= x2);
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.bs.push("(" + stack.fs.pop() + "<=" + stack.fs.pop() + ")");
            }
        protected:
            NodeLEQ* clone_impl() const override { return new NodeLEQ(*this); };  
    };
}	

#endif
