/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SQRT
#define NODE_SQRT

#include "node.h"

namespace FT{
	class NodeSqrt : public Node
    {
    	public:
    	
    		NodeSqrt()
            {
                name = "sqrt";
    			otype = 'f';
    			arity['f'] = 1;
    			arity['b'] = 0;
    			complexity = 2;
    		}
    		
            /// Evaluates the node and updates the stack states. 
            void evaluate(const MatrixXd& X, const VectorXd& y,
                          const std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > &Z, 
			              Stacks& stack)
            {
                stack.f.push(sqrt(abs(stack.f.pop())));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("sqrt(|" + stack.fs.pop() + "|)");
            }
        protected:
            NodeSqrt* clone_impl() const override { return new NodeSqrt(*this); };  
    };
}	

#endif
