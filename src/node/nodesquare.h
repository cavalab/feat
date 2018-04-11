/* FEWTWO
copyright 2017 William La Cava
license: GNU/GPL v3
*/
#ifndef NODE_SQUARE
#define NODE_SQUARE

#include "node.h"

namespace FT{
	class NodeSquare : public Node
    {
    	public:
    	
    		NodeSquare()
    		{
    			name = "^2";
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
                stack.f.push(pow(stack.f.pop(),2));
            }

            /// Evaluates the node symbolically
            void eval_eqn(Stacks& stack)
            {
                stack.fs.push("(" + stack.fs.pop() + "^2)");
            }
        protected:
            NodeSquare* clone_impl() const override { return new NodeSquare(*this); };  
    };
}	

#endif
