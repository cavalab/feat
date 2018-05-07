/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SUBTRACT
#define NODE_SUBTRACT

#include "node.h"

namespace FT{
	class NodeSubtract : public Node
    {
    	public:
    	
    		NodeSubtract()
    		{
    			name = "-";
    			otype = 'f';
    			arity['f'] = 2;
    			arity['b'] = 0;
    			complexity = 1;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
        		ArrayXd x1 = stack.f.pop();
                ArrayXd x2 = stack.f.pop();
                stack.f.push(limited(x1 - x2));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("(" + stack.fs.pop() + "-" + stack.fs.pop() + ")");
            }
        protected:
            NodeSubtract* clone_impl() const override { return new NodeSubtract(*this); };  
    };
}	

#endif
